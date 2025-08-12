#=
    Algorithm: we have a map of nodes to V (solved nodes)
    - x and y lists are sorted to to impose permutation invariance
    Each level maintains list of nodes to solve
    Dynamic programming iterates through the list from back to front
    At each level iterates through all nodes
    The computation at each node consists of
    - Figuring out the children
    - Access value of children using map of solved nodes
    - Compute optimal xi’s and action
    - Update current node’s value
    Forward pass: build/initialize the list of maps in a front-to-back way
    Solve mdps: solve certainty equivalent mdps at leaves and cache them
    Backward pass: solve value functions
=#

include("./bamdp_structs.jl")
include("../../env/instantiations/uchida_mice_abstraction.jl")
include("../../env/instantiations/constants.jl")
include("./../mdp/util.jl")
include("./../mdp/gpi.jl")

using Match
using .UchidaMdp: construct_mdp
using .GeneralizePolicyIteration: general_policy_iter
using .Util: find_nonzero_idxs, one_hot
using Logging
using DataStructures
using LinearAlgebra:dot
using Distributions: Categorical

logger = ConsoleLogger(stdout)


function forward_pass(v::BAMDPValues)
    push!(v.uhstate_list, Set([UnsortedHyperState(v.s₀)]))
    push!(v.hstate_list, Set([v.s₀]))
    
    for l in range(1, v.L-1)

        push!(v.uhstate_list, Set{UnsortedHyperState}())
        for h in v.uhstate_list[l]
            expand_node(v, h, v.uhstate_list[l+1])
        end

        push!(v.hstate_list, Set([perm_inv(h) for h in v.uhstate_list[l+1]]))
        # @info("Level $(l+1): $(length(v.uhstate_list[l+1])) hyperstates, $(length(v.hstate_list[l+1])) unique")
    end

    v.leaf_istates = Set([InfoState(h.x, dynamic_y_update(h.y, h.τ), h.N₁, h.N₀, h.G) for h in v.uhstate_list[end]])
    # @info("$(length(v.leaf_istates)) certainty equivalent MDPs at leaves.")
end


function unpack_soln(soln)
    return soln.mdp, soln.τ_transition, soln.info_transition
end


function heuristic_reward_variable(G, r)
    return round(G*r; digits=3)
end


function heuristic_reward(G, r, N₁, N₀)
    return heuristic_reward_variable(G, r) + (N₁/(N₁+N₀))
end


function adjusted_rate(G, γ, r, K)
    # solves ∑ₖ γᵏ c = ∑ₖ Gₖ rᵢ γᵏ  for c where Gₖ = G₀ (1-rᵢ)ᵏ
    # this gives a reward rate s.t. you consume entire pool G in cert_equiv_horizon steps at rate c
    if K > 100
        c = r*G*(1-γ)/(1-(1-r)*γ)
    else
        γ⁺ = γ*(1-r)
        c = r*G*( (γ-1)/(γ⁺-1) ) * ( (γ⁺^(K+1) - 1)/(γ^(K+1) - 1) )
    end

    return round(c; digits=3)
end


function certainty_equiv_params(p, G, N₁, N₀)

    mode = p.cert_equiv
    γ, r₁, r₂, K, k = p.γ, p.r₁, p.r₂, p.cert_equiv_horizon, p.cert_equiv_scale

    if mode == "leftover"
        confident_reward = heuristic_reward_variable(G, r₂) + N₁/(N₁+N₀)
        cautious_reward = heuristic_reward_variable(G, r₁) + N₁/(N₁+N₀)
        return confident_reward, confident_reward - cautious_reward

    elseif mode == "adjusted"
        if K == 0
            K = ceil(Int, log(0.1)/log(1-r₂))
        end
         
        confident_reward = adjusted_rate(G, γ, r₂, K) + N₁/(N₁+N₀)
        cautious_reward = adjusted_rate(G, γ, r₁, K) + N₁/(N₁+N₀)
        return confident_reward, confident_reward - cautious_reward

    elseif mode =="explicit_k"
        confident_reward = k*heuristic_reward_variable(G, r₂) + N₁/(N₁+N₀)
        cautious_reward = k*heuristic_reward_variable(G, r₁) + N₁/(N₁+N₀)
        return confident_reward, confident_reward - cautious_reward

    else
        return N₁/(N₁+N₀), 0.0
    end
end


function reward_pool_update(s, G, r₁, r₂)
    if s == REWARD_STATE
        return G-heuristic_reward_variable(G, r₂)
    elseif s == CAUTIOUS_REWARD_STATE
        return G-heuristic_reward_variable(G, r₁)
    else
        return G
    end
end


function add_heuristic_bonus(R, h, p)
    R̂ = copy(R)
    R̂[REWARD_STATE] += heuristic_reward(h.G, p.r₂, h.N₁, h.N₀)
    R̂[CAUTIOUS_REWARD_STATE] += heuristic_reward(h.G, p.r₁, h.N₁, h.N₀)
    return R̂
end


function reward_forgetting(p, G)
    G₀ = p.G
    fᵣ = p.fᵣ
    r_forget_type = p.r_forget_type

    if r_forget_type == "linear"
        return G + fᵣ <= G₀ ? G + fᵣ : G
    elseif r_forget_type == "exponential"
        return G + fᵣ*(G₀-G)
    else
        throw(ArgumentError("Invalid reward forgetting type: $r_forget_type."))
    end

end


function next_hyperstates(soln, p, h, a)
    mdp, τ_transition, info_transition = unpack_soln(soln)
    T_ŝ = mdp.T[h.τ, h.s, a, :]
    s⃗_next = find_nonzero_idxs(T_ŝ)
    T_next = T_ŝ[s⃗_next]
    τ_next = τ_transition(h.s, s⃗_next, h.τ)
    x_next, y_next = info_transition(h.s, a, s⃗_next, h.x, h.y, h.τ)
    G_next = fill(reward_pool_update(h.s, h.G, p.r₁, p.r₂), length(s⃗_next))
    return s⃗_next, T_next, τ_next, x_next, y_next, G_next
end


function expand_node(v::BAMDPValues, h::UnsortedHyperState, h_set)
 
    possible_actions = v.dummy_mdp.possible_actions

    for a in possible_actions(h.s)
        # TODO: implement α updating for pCVaR
        s⃗_next, T_next, τ_next, x_next, y_next, G_next = next_hyperstates(v.dummy_mdp, v.params, h, a)

        for (s₊, τ₊, x₊, y₊, G₊) in zip(s⃗_next, τ_next, x_next, y_next, G_next)

            if s₊ == DEAD_STATE
                continue
            end

            h₊ = UnsortedHyperState(s₊, h.α, τ₊, h.l+1, x₊, y₊, h.N₁, h.N₀, G₊)
            push!(h_set, h₊)
        end
    end

end


function get_ω(haz_type, x, y, p)

    h = p.hazard_priors

    ω = @match haz_type begin
        "weibpostmean" => [x, y, h.α, h.β, h.k]
        "betapostmean" => [x, y, h.α, h.β]
        "betamonopostmean" => [x, y, h.α, h.β]
        "noisyorpostmean" => [x, y, h.α, h.β]
        _     => throw(ArgumentError("Invalid hazard type: $haz_type."))
    end

    return ω
end


function solve_certain_mdps(v::BAMDPValues, α_only::Float64)
    p = v.params
    haz_type = p.hazard_type

    for i in collect(v.leaf_istates)

        r₀, Cₓ = certainty_equiv_params(p, i.G, i.N₁, i.N₀)

        mdp = construct_mdp(haz_type=haz_type, ω=get_ω(haz_type, i.x, i.y, p),
        detect_cost=p.Cₒ, retreat_cost=p.Cᵣ, dying_cost=p.Cₜ, γ=p.γ, α⃗=p.α⃗, H=p.H,
        p₁=p.p₁, p₂=p.p₂, caution_cost=Cₓ, r₀=r₀, converge_thresh=p.Δ)

        general_policy_iter(mdp, "value_iter", v.params.I, "n", α_only)
        v.leaf_mdps[i] = mdp.soln
    end

end


function solve_last_level(v::BAMDPValues)
    for h in v.hstate_list[end]
        soln = v.leaf_mdps[InfoState(h.x, dynamic_y_update(h.y, h.τ), h.N₁, h.N₀, h.G)]

        iₐ = findfirst(x->x==h.α, soln.values.α⃗)

        if isnothing(iₐ)
            throw(ErrorException("α=$(h.α) not in α⃗ of certainty solution. Interpolation isn't implemented."))
        end

        v.𝕍[h] = soln.values.𝕍[iₐ, h.τ, h.s]  
    end
end


function optimize_ncvar(input, α)
    cdf = 0.0
    sorted_by_value = sort(input, by=first)
    ξ = []

    for (V, p) in sorted_by_value
        if cdf == 1
            push!(ξ, 0)
        elseif cdf + p/α > 1
            push!(ξ, (1-cdf)/p)
            cdf = 1
        else
            push!(ξ, 1/α)
            cdf += p/α
        end
    end

    V = dot(ξ, [x[1]*x[2] for x in sorted_by_value])
    # invert sorting to restore original indexing
    idxs = sortperm(sortperm(input, by=first))
    return [ξ[i] for i in idxs], V
end


function nCVaR_Bellman(s, α, R, γ, T_next, V_next)
    # Boundary Conditions
    if α == 0  # minimum over next state V at α=0

        V_min = minimum(V_next)
        V = R[s] + γ*V_min
        ξ⃗ = zeros(size(T_next))
        s⃗_min = [i for (i,V) in enumerate(V_next) if V==V_min]

        for i in s⃗_min
            ξ⃗[i] = 1/(T_next[i]*length(s⃗_min))  # this gives uniform T' = ξ⃗*T over min next states            
        end
        
        return V, ξ⃗
    else
        # Non-Boundary Case
        ξ⃗, V = optimize_ncvar([(V_next[i], T_next[i]) for i in range(1, length(V_next))], α)
        V = R[s] + γ*V

        return V, ξ⃗
    end
end


function inference_mdp(v, h)
    p = v.params
    haz_type = p.hazard_type

    return construct_mdp(haz_type=haz_type, ω=get_ω(haz_type, h.x, dynamic_y_update(h.y, h.τ), p), 
     detect_cost=p.Cₒ, retreat_cost=p.Cᵣ,
     dying_cost=p.Cₜ, γ=p.γ, α⃗=p.α⃗, H=p.H, p₁=p.p₁, p₂=p.p₂, caution_cost=p.Cₓ,
     r₀=0.0, converge_thresh=p.Δ).soln
end


function solve_node(v::BAMDPValues, h)
    if haskey(v.𝕍, h)
        return v.𝕍[h]
    end

    soln = inference_mdp(v, h)
    mdp, possible_actions = soln.mdp, soln.possible_actions
    R = add_heuristic_bonus(mdp.R, h, v.params)

    runs_a = []

    for a in range(1, mdp.A)
        if a in possible_actions(h.s)
            s⃗_next, T_next, τ_next, x_next, y_next, G_next = next_hyperstates(soln, v.params, h, a)
            V_next = [s₊ == DEAD_STATE ? 0.0 : v.𝕍[HyperState(s₊, h.α, τ₊, h.l+1, x₊, y₊, h.N₁, h.N₀, G₊)] for (s₊, τ₊, x₊, y₊, G₊) in zip(s⃗_next, τ_next, x_next, y_next, G_next)]
            V, ξ⃗ = nCVaR_Bellman(h.s, h.α, R, v.params.γ, T_next, V_next)
            push!(runs_a, (V, ξ⃗, s⃗_next))
        else
            push!(runs_a, (-Inf, -Inf, -Inf))  # stupid hack to deal with illegal actions       
        end
    end

    # compute max action
    Qs = [run[1] for run in runs_a]
    a_max = argmax(Qs)
    Q_max, ξ⃗, s⃗_next = runs_a[a_max]

    # update values
    v.ℚ[h] = Qs
    v.𝕍[h] = Q_max
    ξ_all = zeros(mdp.S)

    for i in range(1, length(s⃗_next))
        ξ_all[s⃗_next[i]] = ξ⃗[i]
    end

    v.ξᵥ[h] = ξ_all

   # update policy
   a⃗_best = convert(Array{Float64}, abs.(Qs .- maximum(Qs)) .< 1e-3)  # 1e-3 arbitrary small number
   v.π[h] = sum(a⃗_best) == 1 ? one_hot(a_max, mdp.A) : a⃗_best ./ sum(a⃗_best)
end


function backward_pass(v::BAMDPValues, α_only, mthread=false)
    if mthread
        solve_certain_mdps(v, α_only, "./out/leaf_mdps.jls")  # calls method in run.jl which is hacky
    else
        solve_certain_mdps(v, α_only)
    end

    solve_last_level(v)

    for i in reverse(1:v.L-1)
        for h in v.hstate_list[i]
            solve_node(v, h)
        end
    end

end


function plan(p, s₀, α_only=nothing, mthread=false)

    v = BAMDPValues(p, s₀, p.L, 
        construct_mdp(haz_type="uniform", ω=[], detect_cost=-1.5, retreat_cost=-1.5, dying_cost=-5,
         γ=0.9, α⃗=p.α⃗, H=p.H, p₁=0.4, p₂=0.6, caution_cost=0.0, r₀=0.0, converge_thresh=1e-4))

    forward_pass(v)
    backward_pass(v, α_only, mthread)
    return v
end


function online_simulation(p; λ_true)

    α = p.α⃗[1]
    h₀ = HyperState(1, α, 1, 1, [], [], p.N₁, p.N₀, p.G)

    soln = construct_mdp(haz_type="weibull", ω=[2.0, λ_true^2.0],
     detect_cost=p.Cₒ, retreat_cost=p.Cᵣ, dying_cost=p.Cₜ, γ=p.γ, α⃗=p.α⃗, H=p.H,
     p₁=p.p₁, p₂=p.p₂, caution_cost=p.Cₓ, r₀=0.0, converge_thresh=p.Δ).soln

    mdp, τ_transition, info_transition = unpack_soln(soln)
    T, R = mdp.T, mdp.R

    trajectory = [h₀]
    rewards = []
    actions = []

    function forgetting_update(v, s₀)
        
        G = reward_forgetting(v.params, s₀.G)
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

        if s₀.s == DEAD_STATE || length(trajectory) >= p.max_steps
            return "Early Termination"
        end

        push!(actions, a)
        
        T_ŝ = T[s₀.τ, s₀.s, a, :]
        ŝ = rand(Categorical(T_ŝ))  # sample

        # todo: hack, make sure we don't get detected
        if ŝ == DETECT_STATE
            ŝ = REWARD_STATE
        elseif ŝ == CAUTIOUS_DETECT_STATE
            ŝ = CAUTIOUS_REWARD_STATE
        end
        
        τ = τ_transition(s₀.s, [ŝ], s₀.τ)[1]
        x, y = info_transition(s₀.s, a, [ŝ], s₀.x, s₀.y, s₀.τ)
        N₁ = s₀.N₁; N₀ = s₀.N₀+1
        G = reward_pool_update(s₀.s, s₀.G, v.params.r₁, v.params.r₂)
        return HyperState(ŝ, α, τ, 1, x[1], y[1], N₁, N₀, G)
    end

    while true
        h₀ = trajectory[end]
        v = plan(p, h₀, α)
        h⁺ = true_dynamics_update(v, h₀)

        if h⁺ == "Early Termination"
            break
        end

        h⁺ = forgetting_update(v, h⁺)

        push!(trajectory, h⁺)
    end
    
    return (trajectory, rewards, actions)
end