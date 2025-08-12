function α⃗_log(levels, low, high)
    logl, logh = log.([low ; high])
    return vcat(0, round.(exp.(Array(logl:(logh-logl)/(levels-1):logh)); digits=5))
end


function α⃗_uniform(levels, low, high)
    return vcat(0, round.(Array(low:(high-low)/(levels-1):high); digits=5))
end