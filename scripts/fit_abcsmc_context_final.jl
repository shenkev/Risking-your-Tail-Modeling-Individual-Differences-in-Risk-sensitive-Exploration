include("../smc/smc.jl")
include("../smc/models/model_noisyor_final.jl")


using Distributions: Uniform
using .ABCSMC: SMCProblem, SMCHistory, run, save
using JSON: parsefile


intermediate_animals = Set([32])
confident_animals = Set([26, 27, 28, 29, 30, 31, 33, 34])

animalᵢ = parse(Int, ARGS[1])
data = parsefile("./animal_stats/abc_gt_targets_context.json")
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

priors = [
    Uniform(0.4, 1.0),  # cvar
    Uniform(1, 60),  # Tᴳ
    Uniform(0.01, 0.35),  # fᵣ
]

hazard_priors = [Uniform(0.0, 0.5),  # μ₁
                     Uniform(0.0, 0.5),  # μ₂
                     Uniform(0., 0.95),  # μ₃
                     Uniform(0.05, 0.95),  # u₁
                     Uniform(0.05, 0.95),  # u₂
                     Uniform(0.05, 0.95),  # u₃
    ]

priors = vcat(priors, hazard_priors)
ϵ₀ = 0.5

if animalᵢ in intermediate_animals

    animal_group = "intermediate"
    x₂, t₁, t₂, d₁, d₂, f₁, f₂ = data["gt_targets"]["$(animalᵢ-26)"]

    S₀ = [cutoff_animal_to_model(x₂-50.0), duration_animal_to_model(d₁), duration_animal_to_model(d₂), min(f₂/f₁, 1.0)]
    stat_fn = stats_confident_flat
else

    # 1: cautious, 2: confident-peak, 3: confident-ss
    animal_group = "confident"
    x₂, x₃, t₁, t₂, t₃, d₁, d₂, d₃, f₁, f₂, f₃ = data["gt_targets"]["$(animalᵢ-26)"]
    S₀ = [cutoff_animal_to_model(x₂-50.0), cutoff_animal_to_model(x₃-50.0), duration_animal_to_model(d₁), duration_animal_to_model(d₂), duration_animal_to_model(d₃), min(f₃/f₂, 1.0)]

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

run(p, h, ϵ₀, "./out/abcsmc_context_final/animal$(animalᵢ).jls")
