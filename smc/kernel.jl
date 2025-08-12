module TransitionKernel

using StatsBase: cov, AnalyticWeights
using Distributions: MvNormal, Categorical, pdf
using Statistics: mean, std


export MultivariateNormalTransition, fit, sample, pdf_mixture 



function silverman_heuristic(n, d)
    return (4/(n * (d+2) ))^(1.0/(d+4))
end


mutable struct MultivariateNormalTransition

    θ::Matrix{Float64}
    ω::Vector{Float64}
    d::MvNormal
    Σ::Matrix{Float64}
    λ::Float64  # scaling factor

end


function fit(θ, ω, λ=1.0)

    Σ = cov(θ, AnalyticWeights(ω), 2; corrected=true)

    silverman_heuristic
    Nₑ = 1/sum(ω.^2)  # effective sample size
    d = size(Σ)[1]
    bandwidth_factor = silverman_heuristic(Nₑ, d)
 
    Σ = Σ * bandwidth_factor^2 * λ

    return MultivariateNormalTransition(
        θ,
        ω,
        MvNormal(zeros(d), Σ),
        Σ,
        1.0
    )
end


function sample(T::MultivariateNormalTransition, N)

    i = rand(Categorical(T.ω), N)
    θ⁺ = T.θ[:, i] + rand(T.d, N)
    return θ⁺
end


function pdf_mixture(T::MultivariateNormalTransition, θ⁺)
    if typeof(θ⁺) == Vector{Float64}
        return [pdf_one(T, θ⁺)]
    end

    N = size(θ⁺)[2]
    P = zeros(N)
    
    for i in range(1, N)
        P[i] = pdf_one(T, θ⁺[:, i])
    end
    
    return P
end


function pdf_one(T::MultivariateNormalTransition, θ⁺)
    pdf(T.d, θ⁺ .- T.θ)' * T.ω
end


# θ = [[9.0, 11, 18], [4, 5, 6], [1, 1, 1], [10, 11, 12]]
# θ = hcat(θ...)
# ω = [0.5, 0.25, 0.15, 0.10]
# T = fit(θ, ω)
# sample(T, 10000)
# pdf_mixture(θ)
# pdf_mixture([5.0, 6, 6])
# print("done")

end