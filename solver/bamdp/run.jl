include("./treesearch.jl")
include("../../utilities/misc.jl")
include("../../utilities/cvar.jl")


using .Util: state_color, action_color, action_string, size_interp
using .Misc: save_bin, load_bin

using JSON3
using ProgressMeter
using GraphPlot: gplot
using Graphs: SimpleGraph, add_vertices!, add_edge!, ne, nv
using Compose: cm, PDF, draw
using Cairo
using Base.Threads:@threads


function solve_certain_mdps(v::BAMDPValues, α_only::Float64, fpath::String)

    # if isfile(fpath)
    #     storage = load_bin(fpath)

    #     if storage.params == v.params
    #         @info("Loaded certainty equivalent mdps...")
    #         v.leaf_mdps = storage.leaf_mdps
    #         return
    #     else
    #         @warn("""Parameters at file $fpath did not match those of current run.
    #         Loaded: $(storage.params)
    #         Current: $(v.params)
    #         Overwriting cached equivalent mdps...""")
    #     end

    # end

    # @info("Solving certainty equivalent mdps...")

    p = v.params
    haz_type = p.hazard_type

    # progress = Progress(length(v.leaf_istates))

    @threads for i in collect(v.leaf_istates)

        r₀, Cₓ = certainty_equiv_params(p, i.G, i.N₁, i.N₀)

        mdp = construct_mdp(haz_type=haz_type, ω=get_ω(haz_type, i.x, i.y, p),
        detect_cost=p.Cₒ, retreat_cost=p.Cᵣ, dying_cost=p.Cₜ, γ=p.γ, α⃗=p.α⃗, H=p.H,
        p₁=p.p₁, p₂=p.p₂, caution_cost=Cₓ, r₀=r₀, converge_thresh=p.Δ)

        general_policy_iter(mdp, "value_iter", v.params.I, "n", α_only)
        v.leaf_mdps[i] = mdp.soln
        # next!(progress)
    end

    save_bin(CertainMDPStorage(v.params, v.leaf_mdps), fpath)

end

function plot_trajectory(fpath, states)

    running_count = 1
    running_color = states[1]
    colors = []
    counts = []
    for i in range(2, length(states))
        if states[i] == running_color
            running_count += 1
        else
            push!(counts, running_count)
            push!(colors, running_color)
            running_count = 1
            running_color = states[i]
        end
    end

    push!(counts, running_count)
    push!(colors, running_color)

    g = SimpleGraph()
    add_vertices!(g, length(counts))
    for i in range(1, length(counts)-1)
        add_edge!(g, i, i+1)
    end

    function linear_layout(g, max_len=28)
        nlevels = ceil(Int, nv(g)/max_len)
        y_levels = collect(0:1/(nlevels+1):1)[2:end-1]
        x_levels = collect(0:1/(min(nv(g), max_len)-1):1)
        x⃗ = Float64[]
        y⃗ = Float64[]
        for i in range(1, nv(g))
            push!(x⃗, x_levels[((i-1)%max_len)+1])
            push!(y⃗, y_levels[ceil(Int, i/max_len)])
        end
        return (x⃗, y⃗)
    end


    draw(PDF(fpath, 20cm, 20; dpi=42), gplot(g, layout=linear_layout, nodelabel=counts, nodefillc=colors))
end


function print_array_to_file(fpath, a)

    open(fpath*"_G_values.txt", "w") do io

        function prnt(x)
            println(io, [round(y; digits=3) for y in x])
        end

        pairs = (length(a)-1) ÷ 2
        prnt([a[1]])
        for i in range(1, pairs)
            offset = 2*(i-1)
            x, y = a[offset+2], a[offset+3]
            prnt([x, y])
        end
    
        extra_row = mod(length(a)-1, 2)
        if extra_row == 1
            prnt([a[end]])
        end
    end
end

function online_with_visualization(p, soln, h₀, fpath)
# similar to online_simulation() but supports detection events and writes output using GraphPlot module
    α = p.α⃗[1]

    node_labels = []
    edge_labels = []
    node_colors = []
    g = SimpleGraph()
    G⃗ = [h₀.G]

    mdp, τ_transition, info_transition = unpack_soln(soln)
    T, R = mdp.T, mdp.R

    trajectory = [h₀]
    rewards = []
    actions = []

    function forgetting_update(v, s₀)
        
        G = reward_forgetting(v.params, s₀.G)
        push!(G⃗, G)
        return HyperState(s₀.s, s₀.α, s₀.τ, s₀.l, s₀.x, s₀.y, s₀.N₁, s₀.N₀, G)
    end
        
    function true_dynamics_update(v, s₀)
        push!(rewards, R[s₀.s])

        max_τ = size(T)[1]
        if s₀.τ == max_τ
            a = LEAVE_ACTION
        else
            a = argmax(v.π[s₀])
        end

        add_vertices!(g, 1)
        push!(node_colors, state_color(s₀.s))
        opposite_action = Int(a - 1 == 0) + 1
        push!(node_labels, "$(round(v.ℚ[s₀][a]; digits=1))/$(round(v.ℚ[s₀][opposite_action]; digits=1))")
        
        # println("a: $(a)    Q⃗: $(v.ℚ[s₀])")


        if nv(g) > 1
            add_edge!(g, nv(g)-1, nv(g))
            push!(edge_labels, action_string(a))    
        end

        if s₀.s == DEAD_STATE || length(trajectory) >= p.max_steps
            return "Early Termination"
        end

        push!(actions, a)
        
        T_ŝ = T[s₀.τ, s₀.s, a, :]
        ŝ = rand(Categorical(T_ŝ))  # sample
        
        τ = τ_transition(s₀.s, [ŝ], s₀.τ)[1]
        x, y = info_transition(s₀.s, a, [ŝ], s₀.x, s₀.y, s₀.τ)
        N₁ = s₀.N₁; N₀ = s₀.N₀+1
        G = reward_pool_update(s₀.s, s₀.G, v.params.r₁, v.params.r₂)
        push!(G⃗, G)
        return HyperState(ŝ, α, τ, 1, x[1], y[1], N₁, N₀, G)
    end

    while true
        h₀ = trajectory[end]
        v = plan(p, h₀, α, true)
        h⁺ = true_dynamics_update(v, h₀)

        if h⁺ == "Early Termination"
            break
        end

        h⁺ = forgetting_update(v, h⁺)

        push!(trajectory, h⁺)
    end
    
    function linear_layout(g)
        return (collect(0:1/(nv(g)-1):1), fill(0.5, nv(g)))
    end

    node_sizes = fill(1.0, nv(g))
    node_sizes[1] = 2.0  # hack to make nodes smaller

    plot_trajectory(fpath, node_colors)
    print_array_to_file(fpath, G⃗)

    # draw(PDF(fpath, 20cm, 20; dpi=42), gplot(g, layout=linear_layout,
    # nodelabel=node_labels, nodefillc=node_colors, nodesize=node_sizes, nodelabelsize=node_sizes))

    return (trajectory, rewards, actions)
end

function single_run(p, h₀, fpath, N=100; λ_true)

    mkpath(fpath)

    open(fpath*"params.json", "w") do io
        JSON3.write(io, p)
        JSON3.write(io, "λ true dynamics: $λ_true")
    end

    env_true = construct_mdp(haz_type="weibull", ω=[2.0, λ_true^2.0],
     detect_cost=p.Cₒ, retreat_cost=p.Cᵣ, dying_cost=p.Cₜ, γ=p.γ, α⃗=p.α⃗, H=p.H,
     p₁=p.p₁, p₂=p.p₂, caution_cost=p.Cₓ, r₀=0.0, converge_thresh=p.Δ).soln

    samples = []
    
    for i in range(1, N)            
        sample = online_with_visualization(p, env_true, h₀, fpath*"i$(i).pdf")
        push!(samples, sample)
    end

    save_bin(samples, fpath*"online.jls")

    return samples
end

function enumerate_trajectories(v::BAMDPValues)
    incomplete = Stack{Tuple}()
    push!(incomplete, (1.0, [UnsortedHyperState(v.s₀)]))
    complete = []

    while length(incomplete) > 0
        (p, t) = pop!(incomplete)
        hᵤ = t[end]  # unsortd hyperstate
        h = perm_inv(hᵤ)

        soln = inference_mdp(v, h)
        a = argmax(v.π[h])
        s⃗_next, T_next, τ_next, x_next, y_next = next_hyperstates(soln, hᵤ, a)
        ξ_next = v.ξᵥ[h][s⃗_next]

        for (s₊, T₊, τ₊, x₊, y₊) in zip(s⃗_next, T_next.*ξ_next, τ_next, x_next, y_next)
            h₊ = UnsortedHyperState(s₊, h.α, τ₊, h.l+1, x₊, y₊)
            t₊ = (p*T₊, vcat(t, h.s in [DEAD_STATE, RETREAT_STATE, DETECT_STATE] ? "NOTHING" : (a==1 ? "STAY" : "LEAVE"),  h₊))
            
            if h.l+1 == v.L || s₊ == DEAD_STATE
                push!(complete, t₊)
            else
                push!(incomplete, t₊)
            end
        end
    end

    return sort(complete, by=first, rev=true)
end


function bfs_ordered(v::BAMDPValues, min_size=2, max_size=5)

    g = SimpleGraph()
    edges = []
    q = Queue{Tuple}()
    states_visited = [[] for _ in range(1, v.L)]
    enqueue!(q, (1, UnsortedHyperState(v.s₀), "s", [], 0, 0.0, max_size))
    # tuple (l, h, children for action nodes, action, Q-value, node size)
    global y = 1
    global node_index = 2
    global index_offset = 1
    edge_labels = []
    node_labels = []
    node_colors = []
    node_size = []

    while length(q) > 0

        (current_index, hᵤ, type, children, aₚ, Qₐ, n_size) = dequeue!(q)
        add_vertices!(g, 1)

        if hᵤ.l > y
            global y += 1
            global index_offset = 1
        end
        
        h = perm_inv(hᵤ)
        push!(states_visited[y], hᵤ)
        push!(node_colors, state_color(hᵤ.s))
        push!(node_size, n_size)

        if h.l == v.L || h.s == DEAD_STATE
            push!(node_labels, h.s == DEAD_STATE ? 0.0 : round(v.𝕍[h]; digits=2))
            continue
         end

        soln = inference_mdp(v, h)
        a = argmax(v.π[h])
        s⃗_next, T_next, τ_next, x_next, y_next = next_hyperstates(soln, hᵤ, a)
        ξ_next = v.ξᵥ[h][s⃗_next]
        opposite_action = Int(a - 1 == 0) + 1
        push!(node_labels, "$(round(v.ℚ[h][a]; digits=2))/$(round(v.ℚ[h][opposite_action]; digits=2))")

        children = collect(zip(s⃗_next, T_next.*ξ_next, τ_next, x_next, y_next))

        for (s₊, T₊, τ₊, x₊, y₊) in children
            h₊ = UnsortedHyperState(s₊, hᵤ.α, τ₊, hᵤ.l+1, x₊, y₊)
            enqueue!(q, (node_index, h₊, "s", [], 0, 0.0, size_interp(T₊, min_size, max_size)))
            push!(edges, (y, current_index, index_offset))
            push!(edge_labels, action_string(v.π[h]))
            global node_index += 1
            global index_offset += 1
        end

    end

    num_visited = [length(x) for x in states_visited]
    num_visited_accum = [sum(num_visited[1:i]) for i in range(1, length(num_visited))]
    # construct edges
    for e in edges
        (l, parent_index, offset) = e
        add_edge!(g, parent_index, num_visited_accum[l]+offset)
    end

    @info("States visited at each level: $([length(s) for s in states_visited])")
    return states_visited, g, node_labels, edge_labels, node_colors, node_size
end

function bfs_ordered2(v::BAMDPValues, min_size=2, max_size=5)  # this graphs the action nodes too

    g = SimpleGraph()
    edges = []
    q = Queue{Tuple}()
    states_visited = [[] for _ in range(1, 2*v.L-1)]
    enqueue!(q, (1, UnsortedHyperState(v.s₀), "s", [], 0, 0.0, max_size))
    # tuple (l, h, children for action nodes, action, Q-value, node size)
    global prev_node_type = "s"
    global y = 1
    global node_index = 2
    global index_offset = 1
    edge_labels = []
    node_labels = []
    node_colors = []
    node_size = []

    while length(q) > 0

        (current_index, hᵤ, type, children, aₚ, Qₐ, n_size) = dequeue!(q)
        add_vertices!(g, 1)

        if prev_node_type != type
            global y += 1
            global index_offset = 1
            global prev_node_type = type
        end

        if type == "s"
            h = perm_inv(hᵤ)
            push!(states_visited[y], hᵤ)
            push!(node_labels, round(v.𝕍[h]; digits=2))
            push!(node_colors, state_color(hᵤ.s))
            push!(node_size, n_size)
    
            if h.l == v.L
                continue     
             end
    
            soln = inference_mdp(v, h)
            a = argmax(v.π[h])
            s⃗_next, T_next, τ_next, x_next, y_next = next_hyperstates(soln, hᵤ, a)
            ξ_next = v.ξᵥ[h][s⃗_next]
            children = collect(zip(s⃗_next, T_next.*ξ_next, τ_next, x_next, y_next))

            action_order = h.s == REWARD_STATE ? range(1, 2) : reverse(range(1, 2))
            for a̅ in action_order
                if h.s in [RETREAT_STATE, DEAD_STATE] && a̅ != a
                    continue
                end
                
                enqueue!(q, (node_index, hᵤ, "a",
                 a̅ == a ? children : [], a̅, v.ℚ[h][a̅],
                 size_interp(v.π[h][a̅], min_size, max_size)))

                push!(edges, (y, current_index, index_offset))
                global node_index += 1
                global index_offset += 1
            end

        else

            push!(states_visited[y], current_index)
            push!(node_labels, round(Qₐ; digits=2))
            push!(node_colors, action_color(aₚ))
            push!(node_size, n_size)

            for (s₊, T₊, τ₊, x₊, y₊) in children
                h₊ = UnsortedHyperState(s₊, hᵤ.α, τ₊, hᵤ.l+1, x₊, y₊)
                enqueue!(q, (node_index, h₊, "s", [], 0, 0.0, size_interp(T₊, min_size, max_size)))
                push!(edges, (y, current_index, index_offset))
                global node_index += 1
                global index_offset += 1
            end
    
        end

    end

    num_visited = [length(x) for x in states_visited]
    num_visited_accum = [sum(num_visited[1:i]) for i in range(1, length(num_visited))]
    # construct edges
    for e in edges
        (l, parent_index, offset) = e
        add_edge!(g, parent_index, num_visited_accum[l]+offset)
    end

    @info("States visited at each level: $([length(s) for s in states_visited])")
    return states_visited, g, node_labels, edge_labels, node_colors, node_size
end

function graph_summary(fpath, v::BAMDPValues)
    states_visited, g, node_labels, edge_labels, node_colors, node_size = bfs_ordered(v)
    x_positions = collect(0:1/(length(states_visited)-1):1)

    max_nodes = maximum([length(x) for x in states_visited])
    increment = 1/(max_nodes-1)
    y_positions = []
    global lb_min = 0.5

    for s in states_visited
        l = length(s)

        if l == 1
            push!(y_positions, [0.5])
        elseif l == max_nodes
            push!(y_positions, collect(0:increment:1))
        else
            lb = increment*(max_nodes-length(s))/2
            lb = minimum([lb, lb_min])
            ub = 1-lb
            push!(y_positions, collect(lb:(ub-lb)/(l-1):ub+1e-8))  # prevent weird rounding errors +1e-8
            global lb_min = lb
        end
    end

    x_positions = [fill(x_positions[i], length(y_positions[i])) for i in range(1, length(y_positions))]
    x_positions = collect(Iterators.flatten(x_positions))
    y_positions = collect(Iterators.flatten(y_positions))

    function tree_layout(g)
        return (x_positions, y_positions)
    end

    draw(PDF(fpath*"policy_graph.pdf", 120cm, 120cm; dpi=50), gplot(g, layout=tree_layout,
    nodelabelsize=node_size, nodelabel=node_labels, edgelabel=edge_labels, nodefillc=node_colors, nodesize=node_size))
end


function condense_trajectory(t)
    t₋ = []
    for x in t
        if typeof(x) == UnsortedHyperState
            push!(t₋, x.s)
        elseif x == "LEAVE"
            push!(t₋, x[1])
        end
    end
    return join(t₋)
end

function simple_summary(x)
    return("""
        \tV(s₀) = $(round((x["v"].𝕍[x["v"].s₀]); digits=2))
        α=$(x["v"].s₀.α)    \tP(t) = $(round(x["t"][1][1]; digits=2))
        \tt = $(condense_trajectory(x["t"][1][2]))
        \tfinal hyperstate = $(x["t"][1][2][end])
    """)
end

function trajectory_summary(x, N=10)
    s = "α=$(x["v"].s₀.α) \tV(s₀) = $(round((x["v"].𝕍[x["v"].s₀]); digits=2)) \t\t final hyperstate\n"
    for i in range(1, min(N, length(x["t"])))
        s *= "\tP(t) = $(round(x["t"][i][1]; digits=4))\t t = $(condense_trajectory(x["t"][i][2])) \t $(x["t"][i][2][end])\n"
    end
    return s
end

function summary(folder, α⃗)
    out = ""
    tout = ""
    @showprogress for α in reverse(α⃗)
        x = load_bin(folder*"run_$(α).jls")
        if out == ""
            out *= join(unpack(x["v"].params), ",")*"\n"
        end        
        out *= simple_summary(x)
        tout *= trajectory_summary(x, 10)
    end

    open(folder*"summary.txt","w") do io
        print(io, out)
    end

    open(folder*"trajectory_summary.txt","w") do io
        print(io, tout)
    end
end

function main(p, s₀, L, fpath="./out/")
    v = plan(p, s₀)
    save_bin(Dict([("α", s₀.α), ("v", v), ("t", enumerate_trajectories(v))]), fpath*"run_$(s₀.α).jls")
    graph_summary(fpath, v)
end


function scan_α(p, α⃗, L, fpath="./out/bamdp/")

    mkpath(fpath)

    open(fpath*"params.json", "w") do io
        JSON3.write(io, p)
    end

    @showprogress for α in reverse(α⃗)
        s₀ = HyperState(1, α, 1, 1, [], [])
        v = plan(p, s₀)
        t = enumerate_trajectories(v)
        save_bin(Dict([("α", α), ("v", v), ("t", t)]), fpath*"run_$(α).jls")
        graph_summary(fpath*"$(α)_", v)
        bfs_ordered(v)
    end

    summary(fpath, α⃗)
end