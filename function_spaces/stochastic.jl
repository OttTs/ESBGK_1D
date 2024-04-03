struct StochasticFunction <: FunctionSpace
    _mesh::Mesh
end

function (v::StochasticFunction)(x)
    x = x + (rand() - 0.5) * element_size(v._mesh)
    x = clamp(x, limits(v._mesh)...)

    return ((element_index(x, v._mesh), 1),)
end

function (v::StochasticFunction)(x, y)
    return ((element_index(x, v._mesh), 1),)
end

num_dofs(v::StochasticFunction) where N = num_elements(v._mesh)

integ(v::StochasticFunction, i) = 1