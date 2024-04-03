struct Mesh
    num_elements::Int64
    limits::Tuple{Float64, Float64}
end

num_elements(mesh::Mesh) = mesh.num_elements

num_nodes(mesh::Mesh) = mesh.num_elements + 1

element_size(mesh::Mesh) = (limits(mesh)[2] - limits(mesh)[1]) / mesh.num_elements

element_index(x, mesh::Mesh) = clamp(ceil(Int, (x - limits(mesh)[1]) / element_size(mesh)), 1, num_elements(mesh))

function relative_position(x, mesh::Mesh)
    i = element_index(x, mesh)
    ξ = (x - limits(mesh)[1]) / element_size(mesh) - i + 1
    return i, ξ
end

limits(mesh::Mesh) = mesh.limits