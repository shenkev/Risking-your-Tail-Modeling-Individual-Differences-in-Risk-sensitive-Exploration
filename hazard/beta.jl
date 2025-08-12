module BetaFns


export time_indep_mean, seq_mean, ints_to_sum_inthot, ints_to_sum_onehot, noisy_or_increment, seq_var_noisyor, seq_haz_var_noisyor


function beta_mean(a, b)
    return a/(a+b)
end

function beta_var(a, b)
    return (a*b)/( (a+b)^2 * (a+b+1) )
end

function beta_second_moment(a, b)
    return (a*(a+1))/((a+b)*(a+b+1))
end

function time_indep_mean(N_detect, N_no)
    return beta_mean.(N_detect, N_no)
end

function seq_mean(N_detect, N_no)
    ti_mean = time_indep_mean(N_detect, N_no)
    probs = Float64[]
    sum = 0.0
    for p in ti_mean
        sum = sum + (1-sum)*p
        push!(probs, sum)
    end
    return probs
end

function seq_var_noisyor(N_detect, N_no)

    μ = time_indep_mean(N_detect, N_no)
    s² = beta_second_moment.(N_detect, N_no)
    h = seq_mean(N_detect, N_no)

    vars = Float64[]
    sum = 0.0

    for t in range(1, length(μ))
        if t == 1
            sum = s²[t]-μ[t]*μ[t]
        else
            sum = sum*(1-2*μ[t])+s²[t]-(1-h[t-1])*(1-h[t-1])*μ[t]*μ[t]
        end
        push!(vars, sum)
    end
        
    return vars
end

function seq_haz_var_noisyor(N_detect, N_no)

    μ = time_indep_mean(N_detect, N_no)
    s² = beta_second_moment.(N_no, N_detect)

    vars = Float64[]
    s_sum = 1.0
    μ_sum = 1.0

    for t in range(1, length(μ))
        μ_sum = μ_sum*(1-μ[t])
        s_sum = s_sum*(s²[t])
        
        μ_term = 1-μ_sum
        v = s_sum - 1 + 2*μ_term - μ_term*μ_term

        push!(vars, v)
    end
        
    return vars
end

function time_indep_var(N_detect, N_no)
    return beta_var.(N_detect, N_no)
end

function seq_var(N_detect, N_no)
    seq_mean = seq_mean(N_detect, N_no)
    ti_var = time_indep_var(N_detect, N_no)

    vars = Float64[]
    sum = 0.0

    for i in range(1, length(seq_mean))
        w = i == 1 ? 0.0 : seq_mean[i-1]
        sum = sum + (1-w)*ti_var[i]
        push!(vars, sum)
    end
        
    return vars
end

function ints_to_sum_inthot(ints, arr_len)
	# assumes experience at t is experience at all previous times as well
	arr = zeros(arr_len)
	for i in ints
        if i < 1
            continue
        end
		for j in range(1, i)
			arr[j] += 1
		end
	end
	return arr
end

function noisy_or_increment(ints, arr_len)
	arr = zeros(arr_len)
	for i in ints
        if i < 1
            continue
        end
		for j in range(1, i)
			arr[j] += (i-j+1)
		end
	end
	return arr
end

function ints_to_sum_onehot(ints, arr_len)
    # experience at t only counts for t
	arr = zeros(arr_len)
	for i in ints
        if i < 1
            continue
        end
        arr[i] += 1
	end
	return arr
end

end