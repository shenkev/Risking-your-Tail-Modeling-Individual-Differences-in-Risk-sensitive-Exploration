#=
    𝕍: |α⃗| x H x |S| --- V([s, α, τ])
    ℚ: (|α⃗| x H x |S|) x |A| --- Q([s, α, τ], a)
    H: H-1 is max number of steps in reward state before dying deterministically (domain of λ(t))
    |α⃗|: discretizations of α
    ξᵥ: ξ(ŝ|s) for each α, τ --- |α⃗| x H x |S| x |S|
    ξₒ: ξ(ŝ|s,a) for each α, τ --- |α⃗| x H x |S| x |A| x |S|
=#
mutable struct CVaRValues
    𝕍::Array{Float64, 3}
    hᵥ::Array{Array{Float64, 3}, 1}
    ℚ::Array{Float64, 4}
    hₒ::Array{Array{Float64, 4}, 1}  # there's no Q subscript
    α⃗::Array{Float64, 1}
    αs::Int    
    ξᵥ::Array{Float64, 4}
end

construct_CVaRValues = function(mdp, α⃗)
    αs = size(α⃗)[1]
    H, S, A = mdp.H, mdp.S, mdp.A
    𝕍 = zeros(αs, H, S)
    ℚ = zeros(αs, H, S, A)
    ξᵥ = zeros(αs, H, S, S)
    return CVaRValues(𝕍, [𝕍], ℚ, [ℚ], α⃗, αs, ξᵥ)
end