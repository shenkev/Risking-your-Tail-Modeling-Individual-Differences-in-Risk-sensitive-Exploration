module Hazards

include("./beta.jl")

using .BetaFns: time_indep_mean, seq_mean, ints_to_sum_inthot, ints_to_sum_onehot, noisy_or_increment, seq_var_noisyor, seq_haz_var_noisyor
import Distributions: pdf, cdf, Beta
using Match
using Distributions
using LinearAlgebra:dot
using Logging

logger = ConsoleLogger(stdout)


export f, S, λ, σ², d, f_ztit, S_zt,λ_ztit, HazardFunction, noisy_or_haz_var,
        construct_poisson, construct_uniform, construct_poisson_mixture, 
        construct_weibull, construct_weibull_mixture, construct_weibull_postmean, 
        construct_beta_postmean, construct_beta_mono_postmean, construct_noisy_or_postmean

struct HazardFunction
    p::Any
    H::Int
    ω::Tuple  # parameters
end

struct MixtureHazardFunction
    π::Array{Float64, 1}  # mixing proportions
    hs::Array{HazardFunction, 1}
    H::Int
    ω::Tuple  # parameters
end

function construct_poisson_mixture(ω)
    H = ω[1]
    λs = ω[2]
    π = ω[3]
    return MixtureHazardFunction(π, [construct_poisson((H, λ)) for λ in λs], H, ω)
end

function construct_poisson(ω)
    H = ω[1]
    λ = ω[2]
    return HazardFunction(Poisson(λ), H, ω)
end

function construct_uniform(ω)
    H = ω[1]
    return HazardFunction(DiscreteUniform(1, H), H, ω)
end

function construct_weibull(ω)
    H = ω[1]
    return HazardFunction(WeibullFunction((ω[2:end])), H, ω)
end

function construct_weibull_mixture(params)
    H = params[1]
    ωs = params[2]
    π = params[3]
    return MixtureHazardFunction(π, [construct_weibull(tuple(H, ω...)) for ω in ωs], H, params)
end

function construct_weibull_postmean(ω)
    H = ω[1]
    return HazardFunction(WeibullPosteriorMean(ω[2:end]...), H, ω)
end

function construct_beta_postmean(ω)
    H = ω[1]
    return HazardFunction(BetaPosteriorMean(ω[2:end]...), H, ω)
end

function construct_beta_mono_postmean(ω)
    H = ω[1]
    return HazardFunction(BetaMonoGenPosteriorMean(ω[2:end]...), H, ω)
end

function construct_noisy_or_postmean(ω)
    H = ω[1]
    return HazardFunction(NoisyOrPosteriorMean(ω[2:end]...), H, ω)
end

#=
    Implementation of discrete Weibull with the domain shifted for {0, 1, 2, ...} to {1, 2, 3,...}
    PMF: exp[-(x-1)^k/λ]-exp[-x^k/λ]
    CDF: 1-exp[-x^k/λ]
=#

struct WeibullFunction
    ω::Tuple  # parameters (k, λ)
end

function pdf(w::WeibullFunction, t)
    k, λ = w.ω
    return exp(-(t-1)^k/λ)-exp(-t^k/λ)
end

function cdf(w::WeibullFunction, t)
    k, λ = w.ω
    return 1-exp(-t^k/λ)
end

#=
    Implementation of the posterior Weibull mean hazard
    I.e. ∫h(t|λ)P(λ|M)dλ  where M is a list of times died (uncensored) and survived (censored)
    Still have domain shifted for {0, 1, 2, ...} to {1, 2, 3,...}
=#
struct WeibullPosteriorMean
    x::Array{Float64, 1}  # die intances
    y::Array{Float64, 1}  # live instances (censored)
    α::Float64
    β::Float64
    k::Float64
end

struct BetaPosteriorMean
    x::Array{Float64, 1}  # die intances
    y::Array{Float64, 1}  # live instances (censored)
    α::Array{Float64, 1}  # array of N1 statistics for all t
    β::Array{Float64, 1}  # array of N0 statistics for all t
end

struct BetaMonoGenPosteriorMean
    x::Array{Float64, 1}  # die intances
    y::Array{Float64, 1}  # live instances (censored)
    α::Array{Float64, 1}  # array of N1 statistics for all t
    β::Array{Float64, 1}  # array of N0 statistics for all t
end

struct NoisyOrPosteriorMean
    x::Array{Float64, 1}  # die intances
    y::Array{Float64, 1}  # live instances (censored)
    α::Array{Float64, 1}  # array of N1 statistics for all t
    β::Array{Float64, 1}  # array of N0 statistics for all t
end

function beta_post_mean(w::BetaPosteriorMean, t::Int)
    x, y, α, β = w.x, w.y, w.α, w.β
    x = x .- 1  # it's weird but experience of 4 really means hazard at 3rd decision point for stay
    y = y .- 1
    αₓ = ints_to_sum_onehot(Int.(x), length(α)) .+ α
    βₓ = ints_to_sum_onehot(Int.(y), length(α)) .+ β
    return time_indep_mean(αₓ, βₓ)[t]
end

function beta_mono_post_mean(w::BetaMonoGenPosteriorMean, t::Int)
    x, y, α, β = w.x, w.y, w.α, w.β
    x = x .- 1
    y = y .- 1
    αₓ = ints_to_sum_inthot(Int.(x), length(α)) .+ α  # for events that do happen, this is wrong, they happen at t only
    βₓ = ints_to_sum_inthot(Int.(y), length(α)) .+ β  # for αₓ should use ints_to_sum_onehot instead?
    return seq_mean(αₓ, βₓ)[t]
end

function noisy_or_post_update(w::NoisyOrPosteriorMean, t::Int)
    x, y, α, β = w.x, w.y, w.α, w.β
    x = x .- 1
    y = y .- 1
    αₓ = noisy_or_increment(Int.(x), length(α)) .+ α  # for events that do happen, this is wrong, they happen at t only
    βₓ = noisy_or_increment(Int.(y), length(α)) .+ β  # for αₓ should use ints_to_sum_onehot instead?
    return αₓ, βₓ
end

function noisy_or_post_mean(w::NoisyOrPosteriorMean, t::Int)
    αₓ, βₓ = noisy_or_post_update(w, t)
    return seq_mean(αₓ, βₓ)[t]
end

function noisy_or_post_var(w::NoisyOrPosteriorMean, t::Int)
    αₓ, βₓ = noisy_or_post_update(w, t)
    return seq_var_noisyor(αₓ, βₓ)[t]
end

function noisy_or_post_dist(w::NoisyOrPosteriorMean, t::Int)
    N = 10000000
    αₓ, βₓ = noisy_or_post_update(w, t)
    sum = 1.0
    for i = 1:t
        sum = sum .* (1 .- rand(Beta(αₓ[i], βₓ[i]), N))
    end
    return 1 .- sum
end

function noisy_or_post_haz_var(w::NoisyOrPosteriorMean, t::Int)
    αₓ, βₓ = noisy_or_post_update(w, t)
    return seq_haz_var_noisyor(αₓ, βₓ)[t]
end

function sum_gamma_ratio(w::WeibullPosteriorMean, t::Int)
    x, y, α, β, k = w.x, w.y, w.α, w.β, w.k
    
    m = length(x)
    β₁ = β + sum((y.-1).^k)

    D = 0.0
    Dₜ = 0.0

    for j in range(0, 2^m-1)
        δ = digits(j, base=2, pad=m)        
        βⱼ = β₁ + sum((x.-1 + δ).^k)
        β̃ⱼ = βⱼ + t^k - (t-1)^k
        Δ = (β^α)/(βⱼ^α)
        Δₜ = (β^α)/(β̃ⱼ^α)
        sign = (-1)^(sum(δ) % 2)
        # if (Δ < 1e-9) || (Δₜ < 1e-9)
        #     @warn("Underflow risk in computing term β^α/̃β^α: $(minimum(Δ, Δₜ))")
        # end
        D += sign * Δ
        Dₜ += sign * Δₜ
    end
    # TODO: Do we need to check?
    # if (D < 1e-9) || (Dₜ < 1e-9)
    #     @warn("Underflow risk in computing term Dₜ/̃D: $((Dₜ, D))")
    # end
    return 1-Dₜ/D
end

function f(hf::MixtureHazardFunction, t::Int)
    return dot(hf.π, [f(h, t) for h in hf.hs])
end

function S(hf::MixtureHazardFunction, t::Int)
    return dot(hf.π, [S(h, t) for h in hf.hs])
end

function λ(hf::MixtureHazardFunction, t::Int)
    return f(hf, t)/S(hf, t)
end

function f(hf::HazardFunction, t::Int)
    if t == 0
        return 0.0
    end
    @match hf.p begin
        _::Distributions.DiscreteUniform => return pdf(hf.p, t)
        _::Distributions.Poisson{Float64} => return f_ztit(hf, t)
        _::WeibullFunction => return f_it(hf, t)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

function S(hf::HazardFunction, t::Int)
    @match hf.p begin
        _::Distributions.DiscreteUniform => return round(1-cdf(hf.p, t-1); digits=15)  # cdf is returning numbers off by 1e-17
        _::Distributions.Poisson{Float64} => return S_zt(hf, t)
        _::WeibullFunction => return 1-cdf(hf.p, t-1)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

function λ(hf::HazardFunction, t::Int)
    if t == 0
        return 0.0
    elseif t == hf.H
        return 1.0
    end
    @match hf.p begin
        _::Distributions.DiscreteUniform => return round(f(hf, t)/S(hf, t); digits=12) # cdf rounding is causing division error
        _::Distributions.Poisson{Float64} => return λ_ztit(hf, t)
        _::WeibullFunction => return λ_it(hf, t)
        _::WeibullPosteriorMean => return sum_gamma_ratio(hf.p, t)
        _::BetaPosteriorMean => return beta_post_mean(hf.p, t)
        _::BetaMonoGenPosteriorMean => return beta_mono_post_mean(hf.p, t)
        _::NoisyOrPosteriorMean => return noisy_or_post_mean(hf.p, t)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

function σ²(hf::HazardFunction, t::Int)
    if t == 0
        return 0.0
    elseif t == hf.H
        return 0.0
    end
    @match hf.p begin
        _::NoisyOrPosteriorMean => return noisy_or_post_var(hf.p, t)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

function d(hf::HazardFunction, t::Int)  # returns an empirical/Monte Carlo distribution of the posterior
    if t == 0
        return []
    elseif t == hf.H
        return []
    end
    @match hf.p begin
        _::NoisyOrPosteriorMean => return noisy_or_post_dist(hf.p, t)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

function noisy_or_haz_var(hf::HazardFunction, t::Int)
    if t == 0
        return 0.0
    elseif t == hf.H
        return 0.0
    end
    @match hf.p begin
        _::NoisyOrPosteriorMean => return noisy_or_post_haz_var(hf.p, t)
        _     => throw(ArgumentError("Invalid distribution: $(typeof(hf.p))."))
    end
end

# it is infinity truncated
function f_it(hf::HazardFunction, t::Int)
    if t == hf.H
        return S(hf, t)
    else
        return pdf(hf.p, t)
    end
end

function λ_it(hf::HazardFunction, t::Int)
    if (S(hf, t) == 0) && (f_it(hf, t) == 0) 
        return 1.0
    else
        return f_it(hf, t)/S(hf, t)
    end
end

# ztit is zero truncated infinity truncated
function f_ztit(hf::HazardFunction, t::Int)
    if t == hf.H
        return S_zt(hf, t)
    else
        return pdf(hf.p, t)/(1-pdf(hf.p, 0))
    end
end

function S_zt(hf::HazardFunction, t::Int)
    return 1-(cdf(hf.p, t-1)-pdf(hf.p, 0))/(1-pdf(hf.p, 0))
end

function λ_ztit(hf::HazardFunction, t::Int)
    return f_ztit(hf, t)/S_zt(hf, t)
end

end