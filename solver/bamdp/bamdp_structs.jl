using AutoHashEquals
using StructTypes


@auto_hash_equals struct HyperState
    s::Int
    α::Float64
    τ::Int
    l::Int
    x::Array{Int, 1}
    y::Array{Int, 1}
    N₁::Int
    N₀::Int
    G::Float64
    HyperState(s, α, τ, l, x, y, N₁, N₀, G) = new(s, α, τ, l, sort(x), sort(y), N₁, N₀, G)
end


@auto_hash_equals struct UnsortedHyperState
    s::Int
    α::Float64
    τ::Int
    l::Int
    x::Array{Int, 1}
    y::Array{Int, 1}
    N₁::Int
    N₀::Int
    G::Float64
    UnsortedHyperState(s, α, τ, l, x, y, N₁, N₀, G) = new(s, α, τ, l, x, y, N₁, N₀, G)
    UnsortedHyperState(h::HyperState) = new(h.s, h.α, h.τ, h.l, h.x, h.y, h.N₁, h.N₀, h.G)
end


@auto_hash_equals struct InfoState
    x::Array{Int, 1}
    y::Array{Int, 1}
    N₁::Int
    N₀::Int
    G::Float64
    InfoState(x, y, N₁, N₀, G) = new(sort(x), sort(y), N₁, N₀, G)
end


function perm_inv(h::UnsortedHyperState)
    return HyperState(h.s, h.α, h.τ, h.l, h.x, h.y, h.N₁, h.N₀, h.G)
end


function dynamic_y_update(y, τ)

    if τ == 1  # not at object state
        return y

    else
        steps_at_state = τ-1
        steps_survived = steps_at_state
        ŷ = copy(y)

        if steps_survived > 1  # e.g. τ=3 means agent survived 1 turn, y=2 since S(y) = P(Y≥y=2)
            push!(ŷ, steps_survived)
        end
        
        return ŷ
    end
end


abstract type HazardPriors end


@auto_hash_equals struct NoisyOrPriors <: HazardPriors
    α::Array{Float64, 1}
    β::Array{Float64, 1}
    NoisyOrPriors(;α, β) = new(α, β)
end

@auto_hash_equals struct WeibullPriors <: HazardPriors
    k::Float64
    δ::Float64
    α::Float64
    β::Float64
    WeibullPriors(;k, δ, α) = new(k, δ, α, δ^k)
end


@auto_hash_equals struct Params
    hazard_type::String
    hazard_priors::HazardPriors
    Cₒ::Float64
    Cᵣ::Float64
    Cₜ::Float64
    Cₓ::Float64
    p₁::Float64
    p₂::Float64
    r₁::Float64
    r₂::Float64
    N₁::Int
    N₀::Int
    G::Float64
    fᵣ::Float64
    γ::Float64
    I::Int
    α⃗::Array{Float64, 1}
    Δ::Float64
    H::Int
    L::Int
    max_steps::Int
    r_forget_type::String
    cert_equiv::String
    cert_equiv_horizon::Int
    cert_equiv_scale::Float64
    Params(;hazard_type, hazard_priors, Cₒ, Cᵣ, Cₜ, Cₓ, p₁, p₂, r₁, r₂, N₁, N₀, G, fᵣ, γ, I, α⃗, Δ, H, L, max_steps, r_forget_type, cert_equiv, cert_equiv_horizon, cert_equiv_scale) = new(hazard_type, hazard_priors, Cₒ, Cᵣ, Cₜ, Cₓ, p₁, p₂, r₁, r₂, N₁, N₀, G, fᵣ, γ, I, α⃗, Δ, H, L, max_steps, r_forget_type, cert_equiv, cert_equiv_horizon, cert_equiv_scale)
end


struct CertainMDPStorage
    params::Params
    leaf_mdps::Dict
end


mutable struct BAMDPValues
    params::Params
    s₀::HyperState
    L::Int
    𝕍::Dict{HyperState, Float64}
    ℚ::Dict{HyperState, Array{Float64, 1}}
    ξᵥ::Dict{HyperState, Array{Float64, 1}}
    π::Dict{HyperState, Array{Float64, 1}}
    leaf_istates::Set
    hstate_list::Array{Set{HyperState}, 1}
    uhstate_list::Array{Set{UnsortedHyperState}, 1}
    leaf_mdps::Dict
    dummy_mdp
    BAMDPValues(params, s₀, L, env) = new(
        params, s₀, L, 
        Dict(), Dict(), Dict(), Dict(), Set(), [], [], Dict(),
        env.soln
    )
end


StructTypes.StructType(::Type{HyperState}) = StructTypes.Struct()
StructTypes.StructType(::Type{Params}) = StructTypes.Struct()
StructTypes.StructType(::Type{NoisyOrPriors}) = StructTypes.Struct()
StructTypes.StructType(::Type{WeibullPriors}) = StructTypes.Struct()