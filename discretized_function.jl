"""
    DiscretizedFunction(type, space)

Function that lies inside the function space.
It is defined by N basis functions and the corresponding degrees of freedom:
uₕ(x) = ∑ᵢ(ûᵢ * φᵢ(x))

# Arguments
- type: The data type of the function
- space: The function space
"""
struct DiscretizedFunction{T,S}
    _values::Vector{T}
    _space::S

    function DiscretizedFunction(type::Type, space::FunctionSpace)
        values = zeros(type, num_dofs(space))
        return new{type, typeof(space)}(values, space)
    end
end

"""
    (f::DiscretizedFunction)(x)

Calculates the value of f at the position x
"""
(f::DiscretizedFunction)(x) = sum(f._values[i] * φᵢ for (i,φᵢ) in f._space(x))

"""
    with(f, u::DiscretizedFunction, args::DiscretizedFunction...)

Performs calculations with discretized functions.
Returns a new discretized function.

# Usage
w = with(u, v) do uᵢ, vᵢ
    uᵢ + vᵢ
end
"""
function with(f, u::DiscretizedFunction{T}, args::DiscretizedFunction...) where T
    result = DiscretizedFunction(T, u._space)
    for i in eachindex(u._values)
        #println("---")
        #@time u._values[i]
        #@time args[1]#._space
        #@time (v._values[i] for v in args)
        #println("+++")
        result._values[i] = f(u._values[i], (v._values[i] for v in args)...)
    end
    return result
end

"""
    with!(f, u::DiscretizedFunction, args::DiscretizedFunction...)

Performs calculations with discretized functions.
Updates the first given discretized function `u`.

# Usage
with!(w, u, v) do uᵢ, vᵢ
    uᵢ + vᵢ
end
"""
function with!(f, u::DiscretizedFunction, args::DiscretizedFunction...)
    for i in eachindex(u._values)
        u._values[i] = f((v._values[i] for v in args)...)
    end
end

"""
    uniform!(u, value)

Sets a discretized function to a uniform value.
"""
function uniform!(u, value)
    for i in eachindex(u._values)
        u._values[i] = value
    end
end

"""
    project_add!(u::DiscretizedFunction, x, weight)

Projects and adds a delta distribution, defined by its position x and its weight, onto a discretized function
"""
function project_add!(u::DiscretizedFunction, x, weighting)
    Δx = element_size(u._space._mesh) # TODO this should be ∫ φᵢ(x) dx
    for (i, φᵢ) in u._space(x)
        if(i==0)
            println(x)
            println(i)
            println(φᵢ)
        end
        u._values[i] += φᵢ * weighting / Δx
    end
end