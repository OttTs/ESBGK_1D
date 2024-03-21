struct Lagrange <: FunctionSpace # = Linear
    _mesh::Mesh
end

function (v::Lagrange)(x)
    i, ξ = relative_position(x, v._mesh)
    return ((i, 1-ξ), (i+1, ξ))
end

num_dofs(v::Lagrange) = num_nodes(v._mesh)