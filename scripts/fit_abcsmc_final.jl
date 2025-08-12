include("../smc/smc.jl")
include("../smc/models/model_noisyor_final.jl")


using Distributions: Uniform
using .ABCSMC: SMCProblem, SMCHistory, run, save
using JSON: parsefile


timid_animals = Set([18, 13, 8, 25, 21, 9, 4, 2, 15])
intermediate_animals = Set([10, 11, 24, 17, 1, 6])
confident_animals = Set([5, 16, 20, 12, 19, 0, 22, 23, 14, 3, 7])

animalᵢ = parse(Int, ARGS[1])
data = parsefile("./animal_stats/abc_gt_targets.json")
animal_group = nothing
loss_weights = nothing
max_turns = 200  # TODO: remember to change model_mono_animals.jl:model_μv max_steps when you change this
M = 8000

function duration_animal_to_model(x)
    Δ = 0.75
    y = minimum([(5+1)*Δ, x])
    m = maximum([(3*Δ+y)/(2*Δ), 0.0])
    return m
end

function cutoff_animal_to_model(x)
    #  there are max_turns turns equalling 100 minutes so multiply cutoffs by max_turns/100.0
    return x*(max_turns/100.0)
end

if animalᵢ in [10]
    priors = [
        Uniform(0.4, 1.0),  # cvar
        Uniform(1, 100),  # Tᴳ
        Uniform(0.01, 0.5),  # fᵣ
    ]
else
    priors = [
        Uniform(0.4, 1.0),  # cvar
        Uniform(1, 60),  # Tᴳ
        Uniform(0.01, 0.35),  # fᵣ
    ]
end


if animalᵢ in [7, 5, 12, 0, 22]  # (2, 4*)
    hazard_priors = [Uniform(0.0, 0.5),  # μ₁
                     Uniform(0.0, 0.5),  # μ₂
                     Uniform(0.0, 0.5),  # μ₃
                     Uniform(0.4, 0.95),  # u₁
                     Uniform(0.4, 0.95),  # u₂
                     Uniform(0.4, 0.95),  # u₃
    ]
    # loss_weights = [1, 1, 1, 3, 1]  # cutoff, cutoff, duration, duration, frequency

elseif animalᵢ in [14]  # (3, 4*)
    hazard_priors = [Uniform(0.0, 0.5),  # μ₁
                     Uniform(0.0, 0.5),  # μ₂
                     Uniform(0.0, 0.5),  # μ₃
                     Uniform(0.4, 0.95),  # u₁
                     Uniform(0.4, 0.95),  # u₂
                     Uniform(0.4, 0.95),  # u₃
    ]

elseif animalᵢ in [17, 1, 19]  # (2, 3*)
    hazard_priors = [Uniform(0.05, 0.4),  # μ₁
                     Uniform(0.0, 0.3),  # μ₂
                     Uniform(0.5, 1.0),  # μ₃
                     Uniform(0.4, 1.0),  # u₁
                     Uniform(0.1, 0.9),  # u₂
                     Uniform(0.0, 0.2),  # u₃
    ]

elseif animalᵢ in [10]  # (2, 3*)
    hazard_priors = [Uniform(0.0, 0.35),  # μ₁
                     Uniform(0.0, 0.2),  # μ₂
                     Uniform(0.7, 1.0),  # μ₃
                     Uniform(0.25, 1.0),  # u₁
                     Uniform(0.25, 1.0),  # u₂
                     Uniform(0.0, 0.2),  # u₃
    ]

elseif animalᵢ in [20]  # (2, 3*)
    hazard_priors = [Uniform(0.2, 0.35),  # μ₁
                     Uniform(0.0, 0.06),  # μ₂
                     Uniform(0.5, 1.0),  # μ₃
                     Uniform(0.7, 1.0),  # u₁
                     Uniform(0.7, 1.0),  # u₂
                     Uniform(0.0, 0.2),  # u₃
    ]

elseif animalᵢ in [16]  # (3, 3*)
    hazard_priors = [Uniform(0.0, 0.2),  # μ₁
                     Uniform(0.0, 0.4),  # μ₂
                     Uniform(0.5, 1.0),  # μ₃
                     Uniform(0.3, 0.95),  # u₁
                     Uniform(0.3, 0.95),  # u₂
                     Uniform(0.0, 0.2),  # u₃
    ]
else
    hazard_priors = [Uniform(0.0, 1.0),  # μ₁
                     Uniform(0.0, 1.0),  # μ₂
                     Uniform(0.0, 1.0),  # μ₃
                     Uniform(0.0, 1.0),  # u₁
                     Uniform(0.0, 1.0),  # u₂
                     Uniform(0.0, 1.0),  # u₃
    ]
end
priors = vcat(priors, hazard_priors)


if animalᵢ in [13, 18]
    ϵ₀ = 3.0
elseif animalᵢ in [10]
    ϵ₀ = 1.0
else
    ϵ₀ = 0.5
end

if animalᵢ in timid_animals

    animal_group = "timid"
    xₛ, tₚ, tₛ, dₚ, dₛ, fₚ, fₛ = data["gt_targets"]["$animalᵢ"]

    S₀ = [cutoff_animal_to_model(xₛ-50.0), duration_animal_to_model(dₚ), duration_animal_to_model(dₛ), fₛ/fₚ]
    stat_fn = stats_timid

elseif animalᵢ in intermediate_animals

    animal_group = "intermediate"
    x₂, t₁, t₂, d₁, d₂, f₁, f₂ = data["gt_targets"]["$animalᵢ"]

    S₀ = [cutoff_animal_to_model(x₂-50.0), duration_animal_to_model(d₁), duration_animal_to_model(d₂), min(f₂/f₁, 1.0)]
    stat_fn = stats_confident_flat
else

    # 1: cautious, 2: confident-peak, 3: confident-ss
    animal_group = "confident"
    x₂, x₃, t₁, t₂, t₃, d₁, d₂, d₃, f₁, f₂, f₃ = data["gt_targets"]["$animalᵢ"]
    # freq ratio of steady-state/peak < 1.0
    S₀ = [cutoff_animal_to_model(x₂-50.0), cutoff_animal_to_model(x₃-50.0), duration_animal_to_model(d₁), duration_animal_to_model(d₂), duration_animal_to_model(d₃), f₃/f₂]

    stat_fn = stats_confident

end

p = SMCProblem(
    priors,
    model_μv,
    (pred, gt) -> distance(pred, gt, animal_group, loss_weights),
    S₀,
    100,
    stat_fn,
    -1,
    15,
    200,
    M,
    0.3,
    true,
    false
)

h = SMCHistory()

run(p, h, ϵ₀, "./out/abcsmc_final/animal$animalᵢ.jls")
