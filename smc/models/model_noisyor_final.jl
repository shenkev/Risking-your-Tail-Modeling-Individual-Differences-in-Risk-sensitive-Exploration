using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("./")

@everywhere include("../../solver/bamdp/treesearch.jl")
using StatsBase: mean, median, weights


@everywhere s_map = Dict([
    (NEST_STATE, "N"), 
    (REWARD_STATE, "A"),
    (DETECT_STATE, "D"),
    (DEAD_STATE, "DD"),
    (RETREAT_STATE, "R"),
    (CAUTIOUS_REWARD_STATE, "CA")
    ])


function stats_timid(τ)

    duration_peak = []
    duration_ss = []
    period_peak = []
    period_ss = []

    turns = 0
    confident_bout_tol = 1
    one_turn_tol = 1
    ss_reached = false

    cutoff_ss = -1000.0
    error_return = [-1000, -1000.0, -1000.0, -1000.0]

    for i in range(1, length(τ)÷3)
        
        ((Sₙ, Nₙ), (Sₒ, Nₒ), (Sᵣ, Nᵣ)) = τ[(i-1)*3+1:i*3]

        # error checking
        if Sₒ == s_map[REWARD_STATE]
            
            if confident_bout_tol > 0
                confident_bout_tol -= 1
            else
                return error_return  # quit if we get confident_bout_tol+1 number of confident bouts
                # note: this check goes until end of trajectory
            end
        end

        if Nₒ == 1

            if one_turn_tol > 0
                one_turn_tol -= 1
            else
                return error_return
            end
        end

        # check phase
        if Nₙ > 1 && !ss_reached

            # only count peak-to-steadystate if two nest > 1 in a row
            j = i+1

            if j > length(τ)÷3  # last triplet, just set to steady-state
                cutoff_ss = turns
                ss_reached = true

            else

                ((_, N₊), (_, _), (_, _)) = τ[(j-1)*3+1:j*3]

                if N₊ > 1
                    cutoff_ss = turns
                    ss_reached = true
                end

            end

        end

        if !ss_reached
            push!(duration_peak, Nₒ)
            push!(period_peak, Nₙ)
        else
            push!(duration_ss, Nₒ)
            push!(period_ss, Nₙ)
        end
        
        turns += Nₙ + Nₒ + Nᵣ

    end

    if length(duration_peak) == 0 || length(duration_ss) == 0
        return error_return
    end

    # if length(duration_peak) == 0
    #     return error_return
    # end

    # if length(duration_ss) == 0  # steady-state is peak
    #     duration_ss = duration_peak
    #     period_ss = period_peak
    # end

    dur_peak_mean = mean(duration_peak)
    dur_ss_mean = mean(duration_ss)
    freq_ratio = (mean(period_peak) + mean(duration_peak))/(mean(period_ss) + mean(duration_ss))

    return [cutoff_ss, dur_peak_mean, dur_ss_mean, freq_ratio]
end

function stats_confident(τ)

    duration_cau = []
    duration_peak = []
    duration_ss = []
    period_cau = []
    period_peak = []
    period_ss = []

    turns = 0
    confident_bout_tol = 1
    cautious_bout_tol = 2
    one_turn_tol = 1
    conf_reached = false
    ss_reached = false

    cutoff_conf = -1000.0
    cutoff_ss = -1000.0
    error_return = [-1000, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0]
    data_phase = nothing

    for i in range(1, length(τ)÷3)
        
        ((Sₙ, Nₙ), (Sₒ, Nₒ), (Sᵣ, Nᵣ)) = τ[(i-1)*3+1:i*3]

        if Nₒ == 1  # error checking
    
            if one_turn_tol > 0
                one_turn_tol -= 1
            else
                return error_return
            end
        end

        if !conf_reached && !ss_reached  # cautious phase
            
            if Sₒ == s_map[REWARD_STATE]

                # check if this is error
                j = i+1

                if j > length(τ)÷3  # still haven't found confident approach at end of traj
                    return error_return
                end

                ((_, _), (S₊, _), (_, _)) = τ[(j-1)*3+1:j*3]

                if S₊ == s_map[CAUTIOUS_REWARD_STATE]  # confident approach is error
                    if confident_bout_tol > 0
                        confident_bout_tol -= 1
                    else
                        return error_return
                    end

                    data_phase = "cautious"
    
                else  # confident approach is legit if 2 in a row
                    cutoff_conf = turns
                    conf_reached = true

                    data_phase = "peak"
                end

            else
                data_phase = "cautious"

            end

        end

        if conf_reached && !ss_reached  # peak phase, we use if instead of if-elseif to allow 2 phase transitions at once

            if Sₒ == s_map[CAUTIOUS_REWARD_STATE]
                if cautious_bout_tol > 0
                    cautious_bout_tol -= 1
                else
                    return error_return
                end
            end

            if Nₙ > 1

                # only count peak-to-steadystate if two nest > 1 in a row
                j = i+1

                if j > length(τ)÷3  # last triplet, just set to steady-state
                    cutoff_ss = turns
                    ss_reached = true

                    data_phase = "steady"

                else

                    ((_, N₊), (_, _), (_, _)) = τ[(j-1)*3+1:j*3]

                    if N₊ == 1
                        data_phase = "peak"
        
                    else  # steady-state transition is legit if 2 in a row
                        cutoff_ss = turns
                        ss_reached = true
        
                        data_phase = "steady"
                    end

                end

            else
                data_phase = "peak"

            end

        end

        if conf_reached && ss_reached  # ss phase

            if Sₒ == s_map[CAUTIOUS_REWARD_STATE]
                if cautious_bout_tol > 0
                    cautious_bout_tol -= 1
                else
                    return error_return
                end
            end
    
            data_phase = "steady"

        end
 
        if data_phase == "cautious"
            push!(duration_cau, Nₒ)
            push!(period_cau, Nₙ)
        elseif data_phase == "peak"
            push!(duration_peak, Nₒ)
            push!(period_peak, Nₙ)    
        elseif data_phase == "steady"
            push!(duration_ss, Nₒ)
            push!(period_ss, Nₙ)    
        else
            throw(ArgumentError("Phase not being set properly: $data_phase"))
        end

        turns += Nₙ + Nₒ + Nᵣ
        data_phase = nothing

    end

    if length(duration_peak) == 0 || length(duration_ss) == 0
        return error_return
    end

    if length(duration_cau) == 0
        dur_cau_mean = -99  # special code
    else
        dur_cau_mean = mean(duration_cau)
    end

    dur_peak_mean = mean(duration_peak)
    dur_ss_mean = mean(duration_ss)
    freq_ratio = (mean(period_peak) + mean(duration_peak))/(mean(period_ss) + mean(duration_ss))

    return [cutoff_conf, cutoff_ss, dur_cau_mean, dur_peak_mean, dur_ss_mean, freq_ratio]
end


function stats_confident_flat(τ)

    duration_cau = []
    duration_peak = []
    period_cau = []
    period_peak = []

    turns = 0
    confident_bout_tol = 1
    cautious_bout_tol = 2
    # confident_nest_tol = 3
    # confident_nest_tol_reached = false
    cautious_nest_tol = 1
    one_turn_tol = 1
    conf_reached = false
    ss_reached = false

    cutoff_conf = -1000.0
    cutoff_ss = -1000.0
    error_return = [-1000, -1000, -1000.0, -1000.0, -1000.0]
    data_phase = nothing

    for i in range(1, length(τ)÷3)
        
        ((Sₙ, Nₙ), (Sₒ, Nₒ), (Sᵣ, Nᵣ)) = τ[(i-1)*3+1:i*3]

        if Nₒ == 1  # error checking
    
            if one_turn_tol > 0
                one_turn_tol -= 1
            else
                return error_return
            end
        end

        if !conf_reached  # cautious phase

            # # can't have > 1-nest turns in cautious phase
            # if Nₙ > 1
            #     if cautious_nest_tol > 0
            #         cautious_nest_tol -= 1
            #     else
            #         return error_return
            #     end
            # end
            
            if Sₒ == s_map[REWARD_STATE]

                # check if this is error
                j = i+1

                if j > length(τ)÷3  # still haven't found confident approach at end of traj
                    return error_return
                end

                ((_, _), (S₊, _), (_, _)) = τ[(j-1)*3+1:j*3]

                if S₊ == s_map[CAUTIOUS_REWARD_STATE]  # confident approach is error
                    if confident_bout_tol > 0
                        confident_bout_tol -= 1
                    else
                        return error_return
                    end

                    data_phase = "cautious"
    
                else  # confident approach is legit if 2 in a row
                    cutoff_conf = turns
                    conf_reached = true

                    data_phase = "peak"
                end

            else
                data_phase = "cautious"

            end

        end

        if !ss_reached  # peak phase, we use if instead of if-elseif to allow 2 phase transitions at once

            if Nₙ > 1
                # only count peak-to-steadystate if two nest > 1 in a row
                j = i+1
    
                if j > length(τ)÷3  # last triplet, just set to steady-state
                    cutoff_ss = turns
                    ss_reached = true
                else
    
                    ((_, N₊), (_, _), (_, _)) = τ[(j-1)*3+1:j*3]
    
                    if N₊ > 1
                        # steady-state transition is legit if 2 in a row
                        cutoff_ss = turns
                        ss_reached = true
                    end
                end
            end
        end

        if conf_reached  # peak phase, we use if instead of if-elseif to allow 2 phase transitions at once

            if Sₒ == s_map[CAUTIOUS_REWARD_STATE]
                if cautious_bout_tol > 0
                    cautious_bout_tol -= 1
                else
                    return error_return
                end
            end

            # can't have 1-nest turns in confident phase unless it's a 1.0 freq ratio anima
            # if Nₙ == 1
            #     if confident_nest_tol > 0
            #         confident_nest_tol -= 1
            #     else
            #         confident_nest_tol_reached = true
            #     end
            # end

            data_phase = "peak"

        end
 
        if data_phase == "cautious"
            # only push to cautious phase nest durations if not steady-state phase (> 1 turn at nest)
            if !ss_reached
                push!(period_cau, Nₙ)
            end
            push!(duration_cau, Nₒ)
        elseif data_phase == "peak"
            push!(duration_peak, Nₒ)
            push!(period_peak, Nₙ)    
        else
            throw(ArgumentError("Phase not being set properly: $data_phase"))
        end

        turns += Nₙ + Nₒ + Nᵣ
        data_phase = nothing

    end

    if length(duration_cau) == 0 || length(duration_peak) == 0
        return error_return
    end

    # if confident_nest_tol_reached
    #     if mean(period_peak) > 2.0
    #         return error_return
    #     end
    # end

    dur_cau_mean = mean(duration_cau)
    dur_peak_mean = mean(duration_peak)
    freq_ratio = (mean(period_cau) + mean(duration_cau))/(mean(period_peak) + mean(duration_peak))

    return [cutoff_conf, cutoff_ss, dur_cau_mean, dur_peak_mean, freq_ratio]
end


@everywhere function condense_traj(τ)

    running_count = 1
    running_state = τ[1]
    result = []

    for i in range(2, length(τ))
        if τ[i] == running_state
            running_count += 1
        else
            push!(result, (s_map[running_state], running_count))
            running_count = 1
            running_state = τ[i]
        end
    end

    push!(result, (s_map[running_state], running_count))

    return result
end


@everywhere function stretch_exponentiate(u, μ, C=100.0)
    ϵ = (μ-μ.^2) ./ C
    return exp.(log.(ϵ) + u .* (log.(μ-μ.^2) - log.(ϵ)))
end

@everywhere function μv_αβ_transform(μ, v)
    α = -(μ .* (μ .^ 2 .- μ .+ v)) ./ v
    β = ((μ .- 1) .* (μ .^ 2 .- μ .+ v)) ./ v

    return α, β
end

@everywhere function model_μv(p)

    p₁ = 0.125
    p₂ = 0.25
    rᵣ = 1.1
    rₐ = 0.89

    α, Tᴳ, fᵣ = p[1], p[2], p[3]
    μ = [p[4], p[5], p[6]]
    u = [p[7], p[8], p[9]]

    v = stretch_exponentiate(u, μ)
    α_prior, β_prior = μv_αβ_transform(μ, v)

    α⃗ = [α]
    λ_true = 1000000.0
    
    γ = 0.975
    K = 200
    G =Tᴳ*rᵣ
    r₂ = rᵣ/G
    r₁ = r₂*rₐ

    params = Params(hazard_type="noisyorpostmean", hazard_priors=NoisyOrPriors(α=α_prior, β=β_prior), 
        Cₒ=-1.0, Cᵣ=-1.0, Cₜ=-5.0, Cₓ=0.0, p₁=p₁, p₂=p₂,
        r₁=r₁, r₂=r₂, N₁=1, N₀=1, G=G, fᵣ=fᵣ, γ=γ, I=1000, α⃗=α⃗, Δ=1e-3, H=5, L=5, max_steps=200,
        r_forget_type="linear", cert_equiv="adjusted", cert_equiv_horizon=K, cert_equiv_scale=0.5)

    trajectory, rewards, actions = online_simulation(params, λ_true=λ_true)
    return condense_traj([h.s for h in trajectory])
    
end

function normalized_l1(a, b, c)
    return abs((a-b)/c)
end

function cutoff_scaling(y)
    return minimum([30, 10.0+(4/5)*maximum([y-5, 0])])
end

function c2c_cutoff_min(y, min_value)
    return maximum([y, min_value])
end

function period_scaling(y)
    return minimum([20, 2.0+(18/19)*maximum([y-1, 0])])
end

function duration_scaling(y)
    return 2.5  # range is about 2.5 in model space, 3.75 in animal space
end

# function duration_clip(y)
#     Δ = 0.75
#     return minimum([(5+1)*Δ, y])
# end

# function duration_map(x)  # model space to animal space
#     # suppose we have bands: (0, w), (w, 2w), (2w, 3w)
#     # let Δ = w/2, then x⁺ = Δ + 2Δ(x-2)
#     Δ = 0.75
#     return maximum([Δ + 2*Δ*(x-2.0), 0.0])
# end

function distance_verbose(pred, gt, animal_group=nothing, ω=nothing)

    if animal_group == "timid"
        cutoff_ss, dur_peak_mean, dur_ss_mean, freq_ratio = pred
        y_cutoff_ss, y_dur_peak_mean, y_dur_ss_mean, y_freq_ratio = gt

        period_ratio = 1/freq_ratio
        y_period_ratio = 1/y_freq_ratio

        normalization = [cutoff_scaling(y_cutoff_ss),
                         duration_scaling(y_dur_peak_mean),
                         duration_scaling(y_dur_ss_mean),
                         period_scaling(y_period_ratio)]

        pred = [cutoff_ss,
                dur_peak_mean,
                dur_ss_mean,
                period_ratio]
                
        gt = [y_cutoff_ss,
              y_dur_peak_mean,
              y_dur_ss_mean,
              y_period_ratio]

    elseif animal_group == "confident"
        error_return = [-1000, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0]
        cutoff_conf, cutoff_ss, dur_cau_mean, dur_peak_mean, dur_ss_mean, freq_ratio = pred
        y_cutoff_conf, y_cutoff_ss, y_dur_cau_mean, y_dur_peak_mean, y_dur_ss_mean, y_freq_ratio = gt

        # if there is no cautious approach and the groundtruth exists then return maximal loss
        if dur_cau_mean == -99
            if y_dur_cau_mean != 0.0
                cutoff_conf, cutoff_ss, dur_cau_mean, dur_peak_mean, dur_ss_mean, freq_ratio = error_return
            end
        end

        period_ratio = 1/freq_ratio
        y_period_ratio = 1/y_freq_ratio

        if y_dur_cau_mean == 0.0

            normalization = [
                cutoff_scaling(y_cutoff_conf),
                cutoff_scaling(y_cutoff_ss),
                duration_scaling(y_dur_peak_mean),
                period_scaling(y_period_ratio)]

            pred = [
                cutoff_conf,
                cutoff_ss,
                dur_peak_mean,
                period_ratio]

            gt = [
                y_cutoff_conf,
                y_cutoff_ss,
                y_dur_peak_mean,
                y_period_ratio]

        else
            normalization = [
                cutoff_scaling(y_cutoff_conf),
                cutoff_scaling(y_cutoff_ss),
                duration_scaling(y_dur_cau_mean),
                duration_scaling(y_dur_peak_mean),
            #  duration_scaling(y_dur_ss_mean),
                period_scaling(y_period_ratio)]

            pred = [
                cutoff_conf,
                cutoff_ss,
                dur_cau_mean,
                dur_peak_mean,
                # duration_map(dur_ss_mean),
                period_ratio]

            gt = [
                c2c_cutoff_min(y_cutoff_conf, 4),
                y_cutoff_ss,
                y_dur_cau_mean,
                y_dur_peak_mean,
                #   duration_clip(y_dur_ss_mean),
                y_period_ratio]
        end

    elseif animal_group == "intermediate"

        cutoff_conf, cutoff_ss, dur_cau_mean, dur_peak_mean, freq_ratio = pred
        y_cutoff_conf, y_dur_cau_mean, y_dur_peak_mean, y_freq_ratio = gt

        period_ratio = 1/freq_ratio
        y_period_ratio = 1/y_freq_ratio

        if cutoff_ss == -1000.0  # model trajectory never reaches steady-state phase
            normalization = [cutoff_scaling(y_cutoff_conf),
                duration_scaling(y_dur_cau_mean),
                duration_scaling(y_dur_peak_mean),
                period_scaling(y_period_ratio)]

            pred = [cutoff_conf,
                dur_cau_mean,
                dur_peak_mean,
                period_ratio]

            gt = [y_cutoff_conf,
                y_dur_cau_mean,
                y_dur_peak_mean,
                y_period_ratio]

        else
            normalization = [cutoff_scaling(y_cutoff_conf),
                cutoff_scaling(y_cutoff_conf),
                duration_scaling(y_dur_cau_mean),
                duration_scaling(y_dur_peak_mean),
                period_scaling(y_period_ratio)]

            pred = [cutoff_conf,
                cutoff_ss,
                dur_cau_mean,
                dur_peak_mean,
                period_ratio]

            gt = [y_cutoff_conf,
                y_cutoff_conf,
                y_dur_cau_mean,
                y_dur_peak_mean,
                y_period_ratio]
        end

    else
        throw(ArgumentError("Invalid animal group: $animal_group"))
    end

    loss_vector = normalized_l1.(pred, gt, normalization)
    ω = isnothing(ω) ? weights(ones(length(loss_vector))) : weights(ω)
    # print(pred, gt, normalization)
    # print(loss_vector)
    return mean(loss_vector, ω), normalization, loss_vector, ω
end

function distance(pred, gt, animal_group=nothing, ω=nothing)
    d, _, _, _ = distance_verbose(pred, gt, animal_group, ω)
    return d
end