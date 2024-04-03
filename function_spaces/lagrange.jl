struct Lagrange{N} <: FunctionSpace # = Linear
    _mesh::Mesh
end

function (v::Lagrange{1})(x)
    i, ξ = relative_position(x, v._mesh)
    return ((i, 1-ξ), (i+1, ξ))
end

function (v::Lagrange{2})(x)
    i, ξ = relative_position(x, v._mesh)

    i = 2 * i
    ξ = 2 * ξ - 1 # from -1 to 1

    f₁ = ξ * (ξ - 1) / 2
    f₂ = (ξ + 1) * (ξ - 1) / -1
    f₃ = (ξ + 1) * ξ / 2

    return ((i-1, f₁), (i, f₂), (i+1, f₃))
end

num_dofs(v::Lagrange{N}) where N = num_nodes(v._mesh) + (N - 1) * num_elements(v._mesh)

function integ(v::Lagrange{1}, i)
    if (i == 1) || (i == num_dofs(v))
        return 0.5
    else
        return 1
    end
end

function integ(v::Lagrange{2}, i)
    if (i == 1) || (i == num_dofs(v))
        return 1/6
    elseif i % 2 == 0
        return 4/6
    else
        return 2/6
    end
end