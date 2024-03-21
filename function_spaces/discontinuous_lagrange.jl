struct DiscontinuousLagrange <: FunctionSpace # = Constant
    _mesh::Mesh
end

(v::DiscontinuousLagrange)(x) = ((element_index(x, v._mesh), 1),)

num_dofs(v::DiscontinuousLagrange) = num_elements(v._mesh)