include("./util.jl")
include("../../utilities/exceptions.jl")

using Distributions
using LinearAlgebra:dot
using Logging
using .Util: find_nonzero_idxs

logger = ConsoleLogger(stdout)


function cVaR_Bellman(s, iₐ, τ, α⃗, 𝕍ₒ, T_ŝ, R, γ, τ_transition, mode)

    if mode == "p"
        throw(ArgumentError("pCVaR not supported with this file. Please apply the hack of renaming bellman_with_pcvar to bellman.jl..."))
    end

    α = α⃗[iₐ]
    s⃗_next = find_nonzero_idxs(T_ŝ)
    T_next = T_ŝ[s⃗_next]
    τ_next = τ_transition(s, s⃗_next, τ)
    V_next = [𝕍ₒ[:, coord...] for coord in zip(τ_next, s⃗_next)]

    # Boundary Conditions
    # Do we want absorbing state?

    if α == 0  # minimum over next state V at α=0

        V_min = minimum([V_α[iₐ] for V_α in V_next])
        V = R[s] + γ*V_min
        ξ⃗ = zeros(size(s⃗_next))
        s⃗_min = [i for (i,V_α) in enumerate(V_next) if V_α[iₐ]==V_min]

        for i in s⃗_min
            ξ⃗[i] = 1/(T_next[i]*length(s⃗_min))  # this gives uniform T' = ξ⃗*T over min next states            
        end
        return V, ξ⃗, s⃗_next

    elseif α == 1  # regular ∑_x̂ P(x̂)*V(x̂,α=1)

        V = R[s] + γ*dot(T_next, [V_α[iₐ] for V_α in V_next])
        ξ⃗ = ones(size(s⃗_next))
        return V, ξ⃗, s⃗_next

    elseif length(s⃗_next) == 1

        V = R[s] + γ*V_next[1][iₐ]
        ξ⃗ = ones(size(s⃗_next))
        return V, ξ⃗, s⃗_next
    end

    # Non-Boundary Case
    ξ⃗, V = optimize_ncvar([(V_ŝ[iₐ], T_next[i]) for (i, V_ŝ) in enumerate(V_next)], α)

    V = R[s] + γ*V

    return V, ξ⃗, s⃗_next
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