module PolicyModule

include("../../utilities/misc.jl")

using .Misc: check_sum_one

export Policy, init_uniform, construct_policy

#=
    |α⃗| x H x |S| x |A| matrix specifying P(a|s, τ, α)
    τ is the amount of time spent previously in rewarding state, 0-indexed so 1 means 0 turns
=#
struct Policy
    π::Array{Float64, 4}
end

construct_policy = function(π)
    check_policy(π)
    return Policy(π)
end

init_uniform = function(A, dims)
    return fill(1/A, dims) 
end

check_policy = function(π)
    dims = size(π)
    for α in range(1, dims[1])
        for τ in range(1, dims[2])
            check_sum_one(π[α, τ, :, :])
        end
    end
end

end