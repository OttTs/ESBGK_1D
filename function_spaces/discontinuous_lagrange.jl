struct DiscontinuousLagrange{T} <: FunctionSpace # = Constant
    _mesh::Mesh
end

(v::DiscontinuousLagrange{1})(x) = ((element_index(x, v._mesh), 1),)

function (v::DiscontinuousLagrange{2})(x)
    i, ξ = relative_position(x, v._mesh)
    k = 2 * (i - 1)
    return ((k + 1, 1-ξ), (k + 2, ξ))
end

num_dofs(v::DiscontinuousLagrange{T}) where T = T*num_elements(v._mesh)

integ(v::DiscontinuousLagrange{T}, i) where T = 1 / T


#=
struct DiscontinuousLagrange{T} <: FunctionSpace
    nodes::SVector{T, Float64}
    _mesh::Mesh

    function DiscontinuousLagrange(polynomial_degree, mesh)
        N = polynomial_degree + 1
        if N==1
            nodes=SVector{N, Float64}(0.5)
        else # N==2
            nodes=SVector{N, Float64}(0, 1)
        end
        return new{N}(nodes, mesh)
    end
end

(v::DiscontinuousLagrange{1})(x) = ((element_index(x, v._mesh), 1),)

function (v::DiscontinuousLagrange{2})(x)
    i, ξ = relative_position(x, v._mesh)
    k = 2 * (i - 1)
    return ((k + 1, 1-ξ), (k + 2, ξ))
end

num_dofs(v::DiscontinuousLagrange{T}) where T = T*num_elements(v._mesh)
=#


