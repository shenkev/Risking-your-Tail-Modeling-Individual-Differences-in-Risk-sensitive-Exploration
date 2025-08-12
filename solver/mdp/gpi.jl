module GeneralizePolicyIteration

include("./pe.jl")
include("./vi.jl")
include("./util.jl")
include("../../utilities/exceptions.jl")
include("../../utilities/misc.jl")

using Logging
using .Misc: xrepeat
using .Util: check_converge
logger = ConsoleLogger(stdout)

export general_policy_iter


function general_policy_iter(env, algo, iterations, mode, α_only=nothing)
    ev = env.soln.values
    em = env.soln.mdp
    𝕍ₒ = ev.𝕍
    𝕍ₙ = zeros(size(𝕍ₒ))
    ℚ = ev.ℚ
    π = env.soln.policy.π
    ξᵥ = ev.ξᵥ
    α⃗ = ev.α⃗
    hᵥ = ev.hᵥ
    hₒ = ev.hₒ
    T = em.T
    R = em.R
    γ = em.γ
    possible_actions = env.soln.possible_actions
    impossible_state = env.soln.impossible_state
    τ_transition = env.soln.τ_transition
    converge_thresh = env.soln.converge_thresh

    converge = false
    
    if algo == "policy_eval"
        T_algo = ∑ₐT(T, π)  # T_ατsŝ
    elseif algo == "value_iter"
        T_algo = xrepeat(T, length(α⃗)) #  T_ατsaŝ
    end
     
    for i in range(1, iterations)

        # @info("Starting $(mode)CVaR $algo iteration: $i...")

        if algo == "policy_eval"
            pe_sweep(𝕍ₙ, 𝕍ₒ, ξᵥ, α⃗, T_algo, R, γ, impossible_state, τ_transition, mode, α_only)
        elseif algo == "value_iter"
            vi_sweep(𝕍ₙ, 𝕍ₒ, ℚ, π, ξᵥ, α⃗, T_algo, R, γ, possible_actions, impossible_state, τ_transition, mode, α_only)
            push!(hₒ, ℚ)
        else
            throw(NotImplementedError("$algo isn't implemented."))
        end

        push!(hᵥ, 𝕍ₙ)
        Δᵥ = 𝕍ₙ-𝕍ₒ
        converge, ϵ_max, Sϵ_max = check_converge(Δᵥ, converge_thresh)

        𝕍ₒ = copy(𝕍ₙ)  # need to be careful with reference vs value here, can't just assign 𝕍ₒ = 𝕍ₙ
        ev.𝕍 = 𝕍ₒ
        env.soln.iterations = i
        env.soln.ϵ_max = ϵ_max
        env.soln.Sϵ_max = Sϵ_max

        if converge
            env.soln.converged = true
            # @info("""
            # $algo converged in: $i iterations.
            # Max error at state: α=$(α⃗[Sϵ_max[1]]) τ=$(Sϵ_max[2]) s=$(Sϵ_max[3]) ϵ: $ϵ_max.
            # """)
            break
        else
            # @info("""
            # $algo iteration $i completed.
            # Max error at state: α=$(α⃗[Sϵ_max[1]]) τ=$(Sϵ_max[2]) s=$(Sϵ_max[3]) ϵ: $ϵ_max.
            # """)
        end
    end

    if !converge
        @warn("""
            $algo failed to converge after: $iterations iterations.
            Final max error for a state: $(env.soln.ϵ_max)
            ω:$(env.hazard.ω)
            """)
    end
end    

end