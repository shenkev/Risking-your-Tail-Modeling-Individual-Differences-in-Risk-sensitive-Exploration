module Util

using Match


export check_converge, find_nonzero_idxs, one_hot, state_color, action_color, action_string, size_interp


function check_converge(Δ, τ)

    Δ_abs = abs.(Δ)
    ϵ_max = maximum(Δ_abs)
    Sϵ_max = Tuple(argmax(Δ_abs))

    return ϵ_max<τ, ϵ_max, Sϵ_max
end


function find_nonzero_idxs(A)  # return nonzero indices of 1D array A
    x = Int[]
    for i in range(1, length(A))
        if A[i] != 0.0
            push!(x, i)
        end
    end
    return x
end

function one_hot(idx, len)
    x = zeros(len)
    x[idx] = 1.0
    return x
end

function state_color(s)
    return @match s begin
        1 => "lightgray"
        2 => "lightgreen"
        3 => "orange"
        4 => "lightblue"
        5 => "red"
        6 => "darkgreen"
        7 => "darkorange2"
        9 => "pink"
        _     => throw(ArgumentError("Invalid state: $s."))
    end
end

function action_color(a)
    return @match a begin
        1 => "ivory2"
        2 => "lightgray"
        3 => "black"
        _     => throw(ArgumentError("Invalid action: $a."))
    end
end

function action_string(a)
    a = findall(x->x==maximum(a), a)
    if length(a) > 1
        return "I"
    else
        if a[1] == 1
            return "S"
        elseif a[1] == 2
            return "A"
        else
            return "CA"
        end
    end
end

function size_interp(p, mi, ma)
    return mi + p*(ma-mi)
end

end