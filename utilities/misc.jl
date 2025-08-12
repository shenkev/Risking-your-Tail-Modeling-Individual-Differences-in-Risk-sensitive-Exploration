module Misc

include("../env/instantiations/constants.jl")

using Serialization  

export init_3D_array, check_sum_one, xrepeat, findnearest, save_bin, load_bin, rangex, s_map, stretch_exponentiate, μv_αβ_transform

s_map = Dict([
    (NEST_STATE, "nest"), 
    (REWARD_STATE, "object"),
    (DETECT_STATE, "detect"),
    (DEAD_STATE, "dead"),
    (RETREAT_STATE, "retreat"),
    (CAUTIOUS_REWARD_STATE, "cautious_object"),
    (CAUTIOUS_DETECT_STATE, "cautious_detect")
    ])

#=
    assuming ixjxk dimensions, content contains list of array values for each i
    e.g. if (s, a, ŝ) then (a, ŝ) transition arrays for each s
=#

function stretch_exponentiate(u, μ, C=100.0)
    ϵ = (μ-μ.^2) ./ C
    return exp.(log.(ϵ) + u .* (log.(μ-μ.^2) - log.(ϵ)))
end

function μv_αβ_transform(μ, v)
    α = -(μ .* (μ .^ 2 .- μ .+ v)) ./ v
    β = ((μ .- 1) .* (μ .^ 2 .- μ .+ v)) ./ v

    return α, β
end

function rangex(start, finish, step_size)
    return collect(range(start, finish, 1 + ceil(Int, (finish-start)/step_size)))
end

function save_bin(o, fpath::String)
    open(f->serialize(f, o), fpath, "w")
end

function load_bin(fpath)
    return open(deserialize, fpath)
end

init_3D_array = function(dims, content)
    @assert dims[1] == length(content)
    @assert dims[2:3] == size(content[1])

    A = zeros(dims)
    for (i, a) in enumerate(content)
        A[i,:,:] = a
    end

    return A
end

check_sum_one = function(A)
    dims = size(A)
    for a in range(1, dims[1])
        @assert sum(A[a, :]) == 1.0
    end
end

xrepeat = function(A, N)
    A_expanded = reshape(A, (1, size(A)...))
    return repeat(A_expanded, N)
end

function findnearest(A, x)
    return findmin(abs.(A.-x))[2]
end

end