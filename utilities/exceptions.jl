struct ConvergenceError <: Exception
    msg::String
end

struct OptimizationError <: Exception
    msg::String
end

struct NotImplementedError <: Exception
    msg::String
end