mutable struct Particle
    position::Float64
    velocity::SVector{3, Float64}
end

function init_uniform(n, u, T, limits, mₛ, ω)
    Nᵣₑₐₗ = n * (limits[2] - limits[1])
    Nₛᵢₘ = round(Int, Nᵣₑₐₗ / ω)

    particles = Particle[]
    for _ in 1:Nₛᵢₘ
        xₚ = rand() * (limits[2] - limits[1]) + limits[1]
        vₚ = sample_velocity.(u, T, mₛ)
        push!(particles, Particle(xₚ, vₚ))
    end

    return particles
end