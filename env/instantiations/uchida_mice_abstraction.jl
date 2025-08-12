module UchidaMdp

include("./constants.jl")
include("../structures/solution.jl")
include("../../utilities/misc.jl")
include("../../hazard/hazard.jl")

using Match
using .Solution
using .Misc
using .Hazards

export construct_mdp, τ_transition, UchidaMdpParams


S = 9
A = 3
H = 10

struct UchidaMdpParams
    p₁::Float64
    p₂::Float64
    retreat_cost::Float64
    detect_cost::Float64
    caution_cost::Float64
    dying_cost::Float64
    r₀::Float64
    hazard::HazardFunction
    soln::CVaRSolution
    H::Int  # hazard function time limit
end

#=
    Arugments:
    ω::hazard parameters
    H::max turn at object
    p₁::P(die|detected) cautious approach
    p₂::P(die|detected) normal approach
    α⃗::CVaR parameter
    r₀::constant reward at object
=#
function construct_mdp(; haz_type, ω, retreat_cost=-1, detect_cost=-1.0, dying_cost=-5.0, γ=0.9,
     α⃗=α⃗_log(20, 0.001, 1.0), H=10, p₁=0.4, p₂=0.6, caution_cost=-0.1, r₀=1.0, converge_thresh=1e-4)

    ω = tuple(vcat(H-1, ω)...)
    hf = @match haz_type begin
        "uniform" => construct_uniform(ω)
        "poisson" => construct_poisson(ω)
        "poisson mixture" => construct_poisson_mixture(ω)  # e.g. ω=[[1, 8], [0.3, 0.7]]
        "weibull" => construct_weibull(ω)
        "weibpostmean" => construct_weibull_postmean(ω)
        "betapostmean" => construct_beta_postmean(ω)
        "betamonopostmean" => construct_beta_mono_postmean(ω)
        "noisyorpostmean" => construct_noisy_or_postmean(ω)
        _     => throw(ArgumentError("Invalid hazard type: $haz_type."))
    end

    T_stationary = base_transitions(p₁, p₂)
    T = nonstationary_transitions(hf, T_stationary)
    R = base_rewards(detect_cost, retreat_cost, caution_cost, dying_cost, r₀)

    soln = construct_CVaRSoln(A, S, H, T, R, γ, α⃗, converge_thresh,
    possible_actions, impossible_state, τ_transition, info_transition)

    # mask invalid actions
    renormalize_policy(soln.policy.π)
    
    return UchidaMdpParams(p₁, p₂, retreat_cost, detect_cost, caution_cost, dying_cost, r₀, hf, soln, H)
end


function base_rewards(detect_cost, retreat_cost, caution_cost, dying_cost, r₀)
    R = zeros(S)

    R[NEST_STATE] = 0.0
    R[REWARD_STATE] = r₀
    R[CAUTIOUS_REWARD_STATE] = r₀ + caution_cost
    R[DETECT_STATE] = detect_cost
    R[CAUTIOUS_DETECT_STATE] = detect_cost
    R[RETREAT_STATE] = retreat_cost
    R[DYING_STATE] = dying_cost
    R[DEAD_STATE] = 0.0

    return R
end


function base_transitions(p₁, p₂)

    T = zeros(S, A, S)
    T[NEST_STATE, STAY_ACTION, NEST_STATE] = 1.0
    T[NEST_STATE, LEAVE_ACTION, REWARD_STATE] = 1.0
    T[NEST_STATE, LEAVE_CAUTIOUS_ACTION, CAUTIOUS_REWARD_STATE] = 1.0
    T[REWARD_STATE, STAY_ACTION, REWARD_STATE] = 1.0
    T[REWARD_STATE, LEAVE_ACTION, RETREAT_STATE] = 1.0
    T[DETECT_STATE, STAY_ACTION, NEST_STATE] = 1-p₂
    T[DETECT_STATE, STAY_ACTION, DYING_STATE] = p₂
    T[DETECT_STATE, LEAVE_ACTION, NEST_STATE] = 1-p₂
    T[DETECT_STATE, LEAVE_ACTION, DYING_STATE] = p₂
    T[RETREAT_STATE, STAY_ACTION, NEST_STATE] = 1.0
    T[RETREAT_STATE, LEAVE_ACTION, NEST_STATE] = 1.0
    T[DEAD_STATE, STAY_ACTION, DEAD_STATE] = 1.0
    T[DEAD_STATE, LEAVE_ACTION, DEAD_STATE] = 1.0
    T[CAUTIOUS_REWARD_STATE, STAY_ACTION, CAUTIOUS_REWARD_STATE] = 1.0
    T[CAUTIOUS_REWARD_STATE, LEAVE_ACTION, RETREAT_STATE] = 1.0
    T[CAUTIOUS_DETECT_STATE, STAY_ACTION, NEST_STATE] = 1-p₁
    T[CAUTIOUS_DETECT_STATE, STAY_ACTION, DYING_STATE] = p₁
    T[CAUTIOUS_DETECT_STATE, LEAVE_ACTION, NEST_STATE] = 1-p₁
    T[CAUTIOUS_DETECT_STATE, LEAVE_ACTION, DYING_STATE] = p₁
    T[DYING_STATE, STAY_ACTION, DEAD_STATE] = 1.0
    T[DYING_STATE, LEAVE_ACTION, DEAD_STATE] = 1.0

    # define invalid state for debugging simplicity
    T[REWARD_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[DETECT_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[RETREAT_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[DEAD_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[CAUTIOUS_REWARD_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[CAUTIOUS_DETECT_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[DYING_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0
    T[INVALID_STATE, STAY_ACTION, INVALID_STATE] = 1.0
    T[INVALID_STATE, LEAVE_ACTION, INVALID_STATE] = 1.0
    T[INVALID_STATE, LEAVE_CAUTIOUS_ACTION, INVALID_STATE] = 1.0

    return T
 
end


function renormalize_policy(π)
    dims = size(π)
    for i in range(1, dims[1])
        for j in range(1, dims[2])
            for s in range(1, dims[3])
                a⃗ = zeros(A)
                a⃗[possible_actions(s)] .= 1.0
                π[i, j, s, :] = a⃗/sum(a⃗)
            end
        end
    end
end


function possible_actions(s)
    if s == NEST_STATE
        return [STAY_ACTION, LEAVE_ACTION, LEAVE_CAUTIOUS_ACTION]
    else
        return [STAY_ACTION, LEAVE_ACTION]
    end
end


function impossible_state(α, τ, s)
    if (s in [NEST_STATE, DETECT_STATE, DEAD_STATE, 
        RETREAT_STATE, CAUTIOUS_DETECT_STATE]) && τ>1
        return true  # τ will reset to 1 whenever agent returns to nest state
    elseif s==REWARD_STATE && τ==1  # τ should be set to 2 when agent lands in reward state
        return true
    elseif s==CAUTIOUS_REWARD_STATE && τ==1
        return true
    elseif s==INVALID_STATE
        return true
    else
        return false
    end
end


# state 2 (object), action 1 (stay): λ(t) die (state 1), 1-λ(t) safe (state 2)
function nonstationary_transitions(hf, T)

    H = hf.H+1
    T = xrepeat(T, H)

    for τ in range(1, H)
        T[τ, REWARD_STATE, STAY_ACTION, :] = zeros(1, S)
        T[τ, REWARD_STATE, STAY_ACTION, REWARD_STATE] = 1-λ(hf, τ-1)
        T[τ, REWARD_STATE, STAY_ACTION, DETECT_STATE] = λ(hf, τ-1)
        T[τ, CAUTIOUS_REWARD_STATE, STAY_ACTION, CAUTIOUS_REWARD_STATE] = 1-λ(hf, τ-1)
        T[τ, CAUTIOUS_REWARD_STATE, STAY_ACTION, CAUTIOUS_DETECT_STATE] = λ(hf, τ-1)
    end
    
    return T
end


# given (s, τ) compute (ŝ, τ̂ ) for each possible next state
function τ_transition(s, s⃗_next, τ)

    τ_next = Int[]
    for ŝ in s⃗_next
        if s == REWARD_STATE && ŝ == REWARD_STATE
            push!(τ_next, τ+1)
        elseif s == CAUTIOUS_REWARD_STATE && ŝ == CAUTIOUS_REWARD_STATE
            push!(τ_next, τ+1)
        elseif s == NEST_STATE && (ŝ in [REWARD_STATE, CAUTIOUS_REWARD_STATE])
            push!(τ_next, 2)
        else
            push!(τ_next, 1)
        end
    end

    return τ_next
end


function info_transition(s, a, s⃗_next, x, y, τ)

    x_next = []
    y_next = []
    steps_at_state = τ-1
    steps_survived = steps_at_state

    for ŝ in s⃗_next
        x₊ = copy(x)
        y₊ = copy(y)

        if s == REWARD_STATE && a == STAY_ACTION && ŝ == DETECT_STATE
            push!(x₊, steps_at_state)
        elseif s == CAUTIOUS_REWARD_STATE && a == STAY_ACTION && ŝ == CAUTIOUS_DETECT_STATE
            push!(x₊, steps_at_state)
        elseif s == REWARD_STATE && a == LEAVE_ACTION && steps_survived > 1
            push!(y₊, steps_survived)
        elseif s == CAUTIOUS_REWARD_STATE && a == LEAVE_ACTION && steps_survived > 1
            push!(y₊, steps_survived)
        end

        push!(x_next, x₊)
        push!(y_next, y₊)
    end

    return x_next, y_next
end

end