module Solution

using Base: Float64, Bool, Tuple
include("./mdp.jl")
include("./values.jl")
include("./policy.jl")

using .PolicyModule: Policy, init_uniform, construct_policy

export CVaRSolution, construct_CVaRSoln

mutable struct CVaRSolution
    mdp::MDP
    values::CVaRValues
    policy::Policy
    converge_thresh::Float64
    iterations::Int
    converged::Bool
    possible_actions::Function
    impossible_state::Function
    τ_transition::Function
    info_transition::Function
    ϵ_max::Float64
    Sϵ_max::Tuple
end

construct_CVaRSoln = function(A, S, H, T, R, γ, α⃗, thresh,
     possible_actions, impossible_state, τ_transition, info_transition)
    mdp = construct_mdp(A, S, H, T, R, γ)
    values = construct_CVaRValues(mdp, α⃗)
    π = init_uniform(A, (values.αs, H, S, A))
    policy = construct_policy(π)
    return CVaRSolution(mdp, values, policy, thresh, 0, false, possible_actions, impossible_state,
     τ_transition, info_transition, 0.0, ())
end

end