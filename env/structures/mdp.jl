include("../../utilities/misc.jl")

using .Misc: check_sum_one

#=
    Two-state MDP is defined as follows:
        S = {1 (nest), 2 (object)}
        A = {stay, leave}
        H = H-1 is the maximum number of steps in state 2 before hazard will certainly occur (boundary condition)
        T = HxSxAxS Array = T(s, a, ŝ) at some index of time t
        R = S Array = R(s) assumed stationary and also (a, ŝ) independent
            - adding (a, ŝ) makes the CVaR expression more complicated
        γ
=#
struct MDP
    A::Int
    S::Int
    H::Int
    T::Array{Float64, 4}
    R::Array{Float64, 1}
    γ::Float64
end

construct_mdp = function(A, S, H, T, R, γ)
    check_transitions(T)
    return MDP(A, S, H, T, R, γ)
end

check_transitions = function(T)
    dims = size(T)
    for τ in range(1, dims[1])
        for s in range(1, dims[2])
            check_sum_one(T[τ, s, :, :])
        end
    end
end