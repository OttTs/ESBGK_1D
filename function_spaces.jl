abstract type FunctionSpace end

include("function_spaces/lagrange.jl")
include("function_spaces/discontinuous_lagrange.jl")
include("function_spaces/stochastic.jl")


(v::FunctionSpace)(x, y) = v(x)