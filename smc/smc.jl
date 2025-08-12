module ABCSMC


include("./kernel.jl")

using Logging
using Serialization: serialize, deserialize
using LinearAlgebra: normalize
using StatsBase: cov, AnalyticWeights, quantile
using Distributed
using Distributions: Distribution, pdf
using .TransitionKernel

export SMCProblem, SMCHistory, run, save, load


logger = ConsoleLogger(stdout)

#=
    Approximate Bayesian Computation - Sequential Monte Carlo (ABCSMC).

    This is an implementation of an ABCSMC algorithm similar to
    [#tonistumpf]_.

    Parameters
    ----------
    π₀:
        parameter_priors:
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.
    m:
        model:
        θ -> m -> x
    d:
        distance_function:
        Measures the distance of the tentatively sampled particle to the
        measured data.
    x₀:
        ground truth simulation:
    N:
        population_size:
        Specify the size of the population.
        If ``population_specification`` is an ``int``, then the size is
        constant. Adaptive population sizes are also possible by passing a
        :class:`pyabc.populationstrategy.PopulationStrategy` object.
        The default is 100 particles per population.
    S:
        summary_statistics:
        A function which takes the raw model output as returned by
        any ot the models and calculates the corresponding summary
        statistics. Note that the default value is just the identity
        function. I.e. the model is assumed to already calculate
        the summary statistics. However, in a model selection setting
        it can make sense to have the model produce some kind or raw output
        and then take the same summary statistics function for all the models.
    ϵ:
        acceptance threshold:
        a list of floats or -1 for adaptive threshold based on the median of accepted particles
    tₘ:
        max number of populations
    B:
        batch size for candidate proposal:
    M:
        maximum number of candidates to try before giving up and using < N particles
    ϵ_update_quantile:
        quantile of automatic ϵ-update. 0.5=median. Lower is more "aggressive" meaning we try
        to reduce ϵ quickly which means we need to run fewer populations but we risk having fewer
        particles in each population/need to try more candidates
    gt_stats_flag:
        if true then x₀ is already a statistic and we don't need to apply S
    biased:
        use the bias version of smc that upweights particles with lower distance
    self.minimum_epsilon = None
    self.max_nr_populations = None
    self.min_acceptance_rate = None
    self.max_t = None
    self.max_total_nr_simulations = None
    self.max_walltime = None
    self.min_eps_diff = None

=#

struct SMCProblem
    π₀::Array{Distribution, 1}
    m::Function
    d::Function
    x₀::Array
    N::Int
    S::Function
    ϵ::Any
    tₘ::Int
    B::Int
    M::Int
    ϵ_update_quantile::Float64
    gt_stats_flag::Bool
    biased::Bool
end

#=

    History

    Parameters
    ----------
    t:
        population iterator
    θ:
        particles:
    x:
        simulations:
    x₀:
        ground truth, copied from Problem
    ω:
        weights:
    d:
        distances:
    Nₐ:
        N accepted:
    Nₜ
        N rejected:
    ESS
        Effective sample size
=#

mutable struct SMCHistory
    t::Int
    ϵ::Float64
    θ::Array
    x::Array
    x₀::Array
    ω::Array{Array{Float64, 1}, 1}
    d::Array{Array{Float64, 1}, 1}
    Nₐ::Array{Int, 1}
    Nₜ::Array{Int, 1}
    ϵ_used::Array{Float64, 1}
    ESS::Array{Float64, 1}
    SMCHistory() = new(0, 1.0, [], [], [], [], [], [], [], [], [])
end


mutable struct ZeroParticlesException <: Exception
end


function run(problem::SMCProblem, history::SMCHistory, ϵ₀=3.0, fpath="./out/abcsmc_fit.jls")

    history.x₀ = problem.x₀

    if problem.ϵ == -1
        @info("Running with adaptive ϵ, initializing to $ϵ₀")
        history.ϵ = ϵ₀
    else
        @assert length(problem.ϵ) == problem.tₘ
        history.ϵ = problem.ϵ[1]
    end

    while history.t < problem.tₘ

        @time begin
            
            try
                run_generation(problem, history)
                mkpath(dirname(fpath))
                save(history, fpath)
                    
            catch e
                if isa(e, ZeroParticlesException)
                    return
                else
                    throw(e)
                end
                
            end
        end
    end

    return history

end


function sample_prior(problem::SMCProblem, n=100)
    
    θ = [[rand(d) for d in problem.π₀] for _ in range(1, n)]
    # sampled from prior, so all have uniform weight
    ω = ones(length(θ))

    return θ, ω
end


function prior_pdf(π₀, θ)
    
    N = size(θ)[2]
    P = zeros(N)

    function pdf_one(x)
        return prod([pdf(π₀[j], x[j]) for j in range(1, length(π₀))])
    end    

    for i in range(1, N)
        P[i] = pdf_one(θ[:, i])
    end

    return P
end


function sample_posterior(problem::SMCProblem, history::SMCHistory, n=100)

    t = history.t
    θ⁻ = history.θ[t]
    ω⁻ = history.ω[t]


    T = fit(hcat(θ⁻...), ω⁻)  # get transition kernel

    valid_candidates = []
    valid_weights = []

    while length(valid_candidates) < n

        θ⁺ = sample(T, n)

        prior_density = prior_pdf(problem.π₀, θ⁺)
        transition_density = pdf_mixture(T, θ⁺)
        # this is implemented as a single step where you pick from θ of previous round and apply kernel, together

        ω⁺ = prior_density ./ transition_density

        # check for 0 probabilities (lies outside prior range)
        for i in range(1, length(ω⁺))
            if ω⁺[i] > 0.0
                push!(valid_candidates, θ⁺[:, i])
                push!(valid_weights, ω⁺[i])
            end
        end

        # print("\nSampled $(length(valid_candidates)) samples within support.")
        # print("\n$(sum(ω⁺ .== 0.0)) / $(length(ω⁺)) samples lied outside prior support.")
    
    end

    return valid_candidates, valid_weights
end


function propose_candidates(problem::SMCProblem, history::SMCHistory)
    B = problem.B

    if history.t == 0
        return sample_prior(problem, B)
    else
        return sample_posterior(problem, history, B)
    end

end

#=
    Compute the effective sample size of weighted points
    sampled via importance sampling according to the formula

    .. math::
        n_\\text{eff} = \\frac{(\\sum_{i=1}^nw_i)^2}{\\sum_{i=1}^nw_i^2}
=#

function effective_sample_size(ω)
    return sum(ω)^2/sum(ω.^2)
end


function vectorize_mat(A)
    [A[:, i] for i in 1:size(A, 2)]
end


function quantile_ϵ(d, ω, q)
    return quantile(d, AnalyticWeights(ω), q)
end


function median_ϵ(d, ω)
    return quantile_ϵ(d, AnalyticWeights(ω), 0.5)
end


function sample_until_n_accepted(problem::SMCProblem, history::SMCHistory)

    S = problem.S
    N = problem.N
    M = problem.M
    Nₐ = 0
    Nₜ = 0

    t = history.t
    ϵ = history.ϵ

    accepted_particles = []
    accepted_simulations = []
    accepted_weights = Float64[]
    accepted_distances = Float64[]

    while (Nₐ < N) & (Nₜ < M)

        # choose θ
        θ, ω = propose_candidates(problem, history)

        # run model θ -> x using multiprocess
        x = pmap(problem.m, θ)
        # TODO: control number of workers/cores

        # compute distances
        x₀ = problem.gt_stats_flag ? problem.x₀ : S(problem.x₀)
        d⁺ = [problem.d(S(y), x₀) for y in x]

        # compute number accepted
        for i in range(1, length(d⁺))
            if d⁺[i] <= ϵ
                push!(accepted_particles, θ[i])
                push!(accepted_weights, ω[i])
                push!(accepted_simulations, x[i])
                push!(accepted_distances, d⁺[i])
                Nₐ += 1
            end
        end

        Nₜ += length(θ)
    end

    essₜ = round(effective_sample_size(accepted_weights); digits=1)

    @info("Accepted $Nₐ / $Nₜ particles.")
    @info("Acceptance rate: $(round(Nₐ/Nₜ; digits=3))")
    @info("Effective-sample-size: $(essₜ) / $Nₐ")

    push!(history.Nₐ, Nₐ)
    push!(history.Nₜ, Nₜ)
    push!(history.ESS, essₜ)

    return accepted_particles, accepted_simulations, accepted_weights, accepted_distances

end


function ϵbias(d)
    bias_max = 5.0
    percent_worse = (d .- minimum(d))./minimum(d)
    C, D = 5, 1.0  # bias is Cx reduction in weight when percent worse is D%, 1x reduction when percent worse is 0%
    return clamp!(1 .+ (C-1)*percent_worse/D, 1.0, bias_max)
end

function normalize_weights(ω, d, biased)
    if biased
        ω = normalize(ω, 1)
        bias = ϵbias(d)
        return normalize(ω ./ bias, 1)    
    else
        return normalize(ω, 1)
    end
end


function run_generation(problem::SMCProblem, history::SMCHistory)
    t = history.t
    ϵ = history.ϵ

    @info("Starting generation t: $(t+1), ϵ: $(round(ϵ; digits=3)).")

    # perform the sampling
    θ, x, ω, d = sample_until_n_accepted(problem, history)

    if length(θ) == 0
        @warn("0 Particles found in population $(t+1)... ending run.")
        throw(ZeroParticlesException())
    end

    # normalize accepted population weight to 1
    ω = normalize_weights(ω, d, problem.biased)
    @info("Population $(t+1) done.\n")

    # save to history
    push!(history.θ, θ)
    push!(history.x, x)
    push!(history.ω, ω)
    push!(history.d, d)

    # update ϵ
    ϵ_quantile = quantile_ϵ(d, ω, problem.ϵ_update_quantile)
    push!(history.ϵ_used, ϵ_quantile)
    history.ϵ = problem.ϵ == -1 ? ϵ_quantile : problem.ϵ[t+1]

    history.t += 1

end


function save(o::SMCHistory, fpath::String)
    open(f->serialize(f, o), fpath, "w")
end

function load(fpath)
    return open(deserialize, fpath)
end


end