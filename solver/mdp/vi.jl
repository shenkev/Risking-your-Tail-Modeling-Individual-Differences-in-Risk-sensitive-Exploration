include("./util.jl")
include("./bellman.jl")

using LinearAlgebra:dot
using ProgressMeter
using .Util: one_hot


function vi_sweep(𝕍ₙ, 𝕍ₒ, ℚ, π, ξᵥ, α⃗, T_ατsaŝ, R, γ, possible_actions, 
    impossible_state, τ_transition, mode, α_only)

    # @showprogress for s_augmented in CartesianIndices(𝕍ₙ)
    for s_augmented in CartesianIndices(𝕍ₙ)
        iₐ, τ, s = Tuple(s_augmented)
        α = α⃗[iₐ]
        A = size(ℚ)[end]

        if !isnothing(α_only) && α != α_only
            continue
        end

        if impossible_state(α, τ, s)  # save time since some combinations of (τ, s) cannot occur
            continue
        end

        runs_a = Any[]
        for a in range(1, A)
            if a in possible_actions(s)
                T_ŝ = T_ατsaŝ[iₐ,τ,s,a,:]
                Q, ξ⃗, s⃗_next = cVaR_Bellman(s, iₐ, τ, α⃗, 𝕍ₒ, T_ŝ, R, γ, τ_transition, mode)
                ℚ[iₐ, τ, s, a] = Q
                push!(runs_a, (Q, ξ⃗, s⃗_next))
            else
                ℚ[iₐ, τ, s, a] = -Inf
                push!(runs_a, (-Inf, -Inf, -Inf))  # stupid hack to deal with illegal actions
            end
        end

        Qs = [run[1] for run in runs_a]
        a_max = argmax(Qs)
        Q_max, ξ⃗, s⃗_next = runs_a[a_max]
        𝕍ₙ[iₐ, τ, s] = Q_max
        ξᵥ[iₐ, τ, s, s⃗_next] = ξ⃗

       # # TODO uniform tie-break rather than arbitrary
       a⃗_best = convert(Array{Float64}, abs.(Qs .- maximum(Qs)) .< 1e-3)  # 1e-3 arbitrary small number
       if sum(a⃗_best) == 1
            π[iₐ, τ, s, :] = one_hot(a_max, A)
       else
            π[iₐ, τ, s, :] = a⃗_best ./ sum(a⃗_best)
       end

    end
end