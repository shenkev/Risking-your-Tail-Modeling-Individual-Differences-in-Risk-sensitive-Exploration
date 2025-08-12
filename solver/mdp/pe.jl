include("./bellman.jl")

using LinearAlgebra:dot
using ProgressMeter


function pe_sweep(𝕍ₙ, 𝕍ₒ, ξᵥ, α⃗, T_ατsŝ, R, γ, impossible_state, τ_transition, mode, α_only)

    # @showprogress for s_augmented in CartesianIndices(𝕍ₙ)
    for s_augmented in CartesianIndices(𝕍ₙ)
        iₐ, τ, s = Tuple(s_augmented)
        α = α⃗[iₐ]
        T_ŝ = T_ατsŝ[iₐ,τ,s,:]

        if impossible_state(α, τ, s)  # save time since some combinations of (τ, s) cannot occur
            continue
        end

        V, ξ⃗, s⃗_next = cVaR_Bellman(s, iₐ, τ, α⃗, 𝕍ₒ, T_ŝ, R, γ, τ_transition, mode)
        𝕍ₙ[iₐ, τ, s] = V
        ξᵥ[iₐ, τ, s, s⃗_next] = ξ⃗
    end
end


#=  T: H x |S| x |A| x |S| tensor for T(s, a, ŝ) at some index of time τ
    π: |α⃗| x H x |S| x |A| tensor for P(a|s, τ, α)
    output |α⃗| x H x |S| x |S| by summing over a of P(s,a,ŝ) = ∑ₐπ(a|s)T(ŝ|a,s) for each τ,α
=#
function ∑ₐT(T, π)
    αs, H, S, A = size(π)
    T_ατsŝ = zeros(αs, H, S, S)

    for x in CartesianIndices((1:αs, 1:H, 1:S))
        iₐ, τ, s = Tuple(x)
        T_ατs = T[τ, s, :, :]'*π[iₐ, τ, s, :]  # can speed up by removing loop over s
        T_ατsŝ[iₐ, τ, s, :] = T_ατs
    end
    
    return T_ατsŝ
end


#=  R = SxAxS Array = R(s,a,ŝ)
    π: |α⃗| x H x |S| x |A| tensor for P(a|s, τ, α)
    output R(α,τ,s) by summing over a,ŝ of,
    E[R(s,a,ŝ)] = ∑ₐₛ'P(s,a,ŝ)R(s,a,ŝ) = ∑ₐₛ'π(a|s)T(ŝ|a,s)R(s,a,ŝ)
    output |α⃗| x H x |S|
=#
# function ∑ₐₛR(R, T, π)
#     αs, H, S, A = size(π)
#     R_ατs = zeros(αs, H, S)

#     for x in CartesianIndices((1:αs, 1:H, 1:S))
#         iₐ, τ, s = Tuple(x)
#         R_sa = sum(T[τ, s, :, :].*R[s, :, :], dims=2)  # can speed up by removing loop over s
#         R_ατs[iₐ, τ, s] = dot(R_sa, π[iₐ, τ, s, :])
#     end

#     return R_ατs
# end