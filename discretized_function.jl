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

(f::DiscretizedFunction)(x,y) = sum(f._values[i] * φᵢ for (i,φᵢ) in f._space(x,y))

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
    # Δx = element_size(u._space._mesh) # TODO this should be ∫ φᵢ(x) dx
    for (i, φᵢ) in u._space(x, 0)
        #
        u._values[i] += φᵢ * weighting / integ(u._space, i)# / Δx
    end
end

function reconstruct!(u::DiscretizedFunction{T, DiscontinuousLagrange{2}}) where T

    μ = (u._values[1:2:end] + u._values[2:2:end]) / 2
    s = μ[2:end] - μ[1:end-1]

    for i in 1:num_elements(u._space._mesh)
        if i == 1
            m = s[i]
        elseif i == num_elements(u._space._mesh)
            m = s[end]
        else
            m = minmod.(s[i-1], s[i])
        end

        u._values[2 * i - 1] = μ[i] - m / 2
        u._values[2 * i] = μ[i] + m / 2
    end
end

function minmod(a, b)
    a*b < 0 && return zero(a)
    if abs(a) < abs(b)
        return a
    else
        return b
    end
end