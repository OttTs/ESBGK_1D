module ESBGK_1D3V

using LinearAlgebra: cholesky, I, diag, tr, Symmetric
using StaticArrays: SVector, SMatrix
using ProgressMeter

const Kᴮ = 1.380649E-23

include("mesh.jl")
include("function_spaces.jl")
include("discretized_function.jl")

include("particles.jl")
include("species.jl")

include("boundaries.jl")
include("movement.jl")


function simulate!(particles, species, time_step, end_time, mesh, boundaries; conservation_mesh=mesh, function_space=Lagrange(mesh))
    num_steps = floor(Int, end_time / time_step)

    progress = Progress(num_steps)

    for _ in 1:num_steps
        for particle in particles
            move!(particle, species, mesh, boundaries, time_step)
        end

        #----------------------------------------------------------------------------------
        # Calculate parameters for Particle relaxation
        ∑v⁰, ∑v¹, ∑v² = sample_moments(particles, function_space)

        velocity = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
            return ∑v¹ᵢ / ∑v⁰ᵢ
        end

        temperature = with(∑v⁰, ∑v², velocity) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
            ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
            return species.mass / (3 * Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
        end

        relaxation_probability = with(∑v⁰, temperature) do ∑v⁰ᵢ, Tᵢ
            n = species.weighting / element_size(mesh) * ∑v⁰ᵢ
            p = n * Kᴮ * Tᵢ
            μ = dynamic_viscosity(Tᵢ, species.ref_viscosity, species.ref_temperature, species.ref_exponent)
            τ = 1 / species.prandtl_number * μ / p
            return 1 - exp(-time_step / τ)
        end

        transformation_matrix = with(∑v², ∑v⁰, velocity, temperature) do ∑v²ᵢ, ∑v⁰ᵢ, uᵢ, Tᵢ
            Σ = (∑v²ᵢ - ∑v⁰ᵢ * uᵢ * uᵢ') / (∑v⁰ᵢ - 1)
            ν = 1 - 1 / species.prandtl_number
            𝓐 = ((1 - ν) * tr(Σ) / 3 * I + ν * Σ)
            return cholesky(Symmetric(𝓐)).L
        end

        #----------------------------------------------------------------------------------
        # u and T for Conservation stuff
        ∑v⁰, ∑v¹, ∑v² = sample_moments(particles, DiscontinuousLagrange(conservation_mesh))

        u = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
            return ∑v¹ᵢ / ∑v⁰ᵢ
        end

        T = with(∑v⁰, ∑v², u) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
            ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
            return species.mass / (3 * Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
        end

        #----------------------------------------------------------------------------------
        # Relaxation part
        for particle in particles
            relax!(particle, velocity, relaxation_probability, transformation_matrix)
        end

        #----------------------------------------------------------------------------------
        # New u and T for Conservation stuff
        ∑v⁰, ∑v¹, ∑v² = sample_moments(particles, DiscontinuousLagrange(conservation_mesh))

        uₙ = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
            return ∑v¹ᵢ / ∑v⁰ᵢ
        end

        Tₙ = with(∑v⁰, ∑v², uₙ) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
            ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
            return species.mass / (3 * Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
        end

        #----------------------------------------------------------------------------------
        # Conserve momentum and energy
        for particle in particles
            x = particle.position
            particle.velocity = u(x) + √(T(x) / Tₙ(x)) * (particle.velocity - uₙ(x))
        end

        next!(progress)
    end

    finish!(progress)
end

dynamic_viscosity(T, μᵣ, Tᵣ, ω) = μᵣ * (T / Tᵣ)^ω

function sample_moments(particles, function_space)
    ∑v⁰ = DiscretizedFunction(Float64, function_space)
    ∑v¹ = DiscretizedFunction(SVector{3,Float64}, function_space)
    ∑v² = DiscretizedFunction(SMatrix{3,3,Float64,9}, function_space)

    for particle in particles
        project_add!(∑v⁰, particle.position, 1)
        project_add!(∑v¹, particle.position, particle.velocity)
        project_add!(∑v², particle.position, particle.velocity * particle.velocity')
    end

    # Boundary handling...
    if typeof(∑v⁰._space) == Lagrange
        ∑v⁰._values[1] *= 2
        ∑v¹._values[1] *= 2
        ∑v²._values[1] *= 2
        ∑v⁰._values[end] *= 2
        ∑v¹._values[end] *= 2
        ∑v²._values[end] *= 2
    end

    return ∑v⁰, ∑v¹, ∑v²
end

function relax!(particle, velocity, relaxation_probability, transformation_matrix)
    x = particle.position

    rand() > relaxation_probability(x) && return nothing

    u = velocity(x)
    𝓢 = transformation_matrix(x)

    particle.velocity = u + 𝓢 * randn(SVector{3, Float64})
end

most_probable_velocity(T, mₛ) = √(2 * Kᴮ * T / mₛ)

sample_velocity(u, T, mₛ) = u + √0.5 * most_probable_velocity(T, mₛ) * randn()

end