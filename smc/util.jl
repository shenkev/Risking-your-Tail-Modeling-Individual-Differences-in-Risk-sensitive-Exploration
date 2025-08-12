module Util

include("../env/instantiations/constants.jl")

export condense_traj

s_map = Dict([
    (NEST_STATE, "N"), 
    (REWARD_STATE, "A"),
    (DETECT_STATE, "D"),
    (DEAD_STATE, "DD"),
    (RETREAT_STATE, "R"),
    (CAUTIOUS_REWARD_STATE, "CA")
    ])

function condense_traj(τ)

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

end