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


function simulate!(particles, species, time_step, end_time, mesh, boundaries; conservation_mesh=mesh, function_space=Lagrange(mesh), do_USP=false)
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

        if do_USP
            σ = with(∑v², ∑v⁰, velocity) do ∑v²ᵢ, ∑v⁰ᵢ, uᵢ
                Σ = (∑v²ᵢ - ∑v⁰ᵢ * uᵢ * uᵢ') / (∑v⁰ᵢ - 1)
                return Σ - tr(Σ) / 3 * I
            end

            q = sample_heatflux(particles, ∑v⁰, ∑v¹, function_space)

            τ = with(∑v⁰, temperature) do ∑v⁰ᵢ, Tᵢ
                n = species.weighting / element_size(mesh) * ∑v⁰ᵢ
                p = n * Kᴮ * Tᵢ
                μ = dynamic_viscosity(Tᵢ, species.ref_viscosity, species.ref_temperature, species.ref_exponent)
                return μ / p
            end

            p = with(∑v⁰, temperature) do ∑v⁰ᵢ, Tᵢ
                n = species.weighting / element_size(mesh) * ∑v⁰ᵢ
                return n * Kᴮ * Tᵢ
            end
        else
            transformation_matrix = with(∑v², ∑v⁰, velocity, temperature) do ∑v²ᵢ, ∑v⁰ᵢ, uᵢ, Tᵢ
                Σ = (∑v²ᵢ - ∑v⁰ᵢ * uᵢ * uᵢ') / (∑v⁰ᵢ - 1)

                n = species.weighting / element_size(mesh) * ∑v⁰ᵢ
                p = n * Kᴮ * Tᵢ

                μ = dynamic_viscosity(Tᵢ, species.ref_viscosity, species.ref_temperature, species.ref_exponent)
                τ = 1 / species.prandtl_number * μ / p

                γ = (1 - exp(-time_step / τ)) / (time_step / τ)
                Pr = species.prandtl_number
                Σ = (γ * Σ + (1 - γ) * tr(Σ) / (3 * Pr) * I) / (1 / Pr + γ * (1 - 1 / Pr))

                #Σ = I(3) .* Σ + (1 - exp(-time_step / τ)) / (time_step / τ) * (ones(SMatrix{3,3,Float64,9}) - I) .* Σ

                ν = 1 - 1 / species.prandtl_number
                𝓐 = (1 - ν) * tr(Σ) / 3 * I + ν * Σ

                return cholesky(Symmetric(𝓐)).L
            end
        end

        #----------------------------------------------------------------------------------
        # u and T for Conservation stuff
        ∑v⁰, ∑v¹, ∑v² = sample_moments(particles, DiscontinuousLagrange{1}(conservation_mesh))

        u = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
            return ∑v¹ᵢ / ∑v⁰ᵢ
        end

        T = with(∑v⁰, ∑v², u) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
            ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
            return species.mass / (3 * Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
        end

        #----------------------------------------------------------------------------------
        # Relaxation part
        if do_USP
            for particle in particles
                relax!(particle, relaxation_probability, velocity, p, temperature, σ, q, τ, time_step, species)
            end
        else
            for particle in particles
                relax!(particle, velocity, relaxation_probability, transformation_matrix)
            end
        end

        #----------------------------------------------------------------------------------
        # New u and T for Conservation stuff
        ∑v⁰, ∑v¹, ∑v² = sample_moments(particles, DiscontinuousLagrange{1}(conservation_mesh))

        uₙ = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
            return ∑v¹ᵢ / ∑v⁰ᵢ
        end

        Tₙ = with(∑v⁰, ∑v², uₙ) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
            ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
            return species.mass / (3 * Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
        end

        #println("T=", round.(sum(T._values ./ Tₙ._values)/25; digits=5))
        #println("u=", u._values ./ uₙ._values)
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
    #if typeof(∑v⁰._space) == Lagrange
    #    #∑v⁰._values[1] *= 2
    #    #∑v¹._values[1] *= 2
    #    #∑v²._values[1] *= 2
    #    #∑v⁰._values[end] *= 2
    #    #∑v¹._values[end] *= 2
    #    #∑v²._values[end] *= 2
    #    ∑v⁰._values[1] = 2 * ∑v⁰._values[2] - ∑v⁰._values[3]
    #    ∑v¹._values[1] = 2 * ∑v¹._values[2] - ∑v¹._values[3]
    #    ∑v²._values[1] = 2 * ∑v²._values[2] - ∑v²._values[3]
#
    #    ∑v⁰._values[end] = 2 * ∑v⁰._values[end-1] - ∑v⁰._values[end-2]
    #    ∑v¹._values[end] = 2 * ∑v¹._values[end-1] - ∑v¹._values[end-2]
    #    ∑v²._values[end] = 2 * ∑v²._values[end-1] - ∑v²._values[end-2]
    #end

    if typeof(function_space) == DiscontinuousLagrange{2}
        reconstruct!(∑v⁰)
        reconstruct!(∑v¹)
        reconstruct!(∑v²)
    end

    return ∑v⁰, ∑v¹, ∑v²
end

function sample_heatflux(particles, ∑v⁰, ∑v¹, function_space)
    ∑cᵢcⱼcⱼ = DiscretizedFunction(SVector{3,Float64}, function_space)

    u = with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
        return ∑v¹ᵢ / ∑v⁰ᵢ
    end

    for particle in particles
        c = particle.velocity - u(particle.position,0) # TODO Interpolieren ?
        project_add!(∑cᵢcⱼcⱼ, particle.position, c * (c' * c))
    end

    return with(∑cᵢcⱼcⱼ, ∑v⁰) do ∑cᵢcⱼcⱼᵢ, ∑v⁰ᵢ
        0.5 * ∑v⁰ᵢ / ((∑v⁰ᵢ - 1) * (∑v⁰ᵢ - 2)) * ∑cᵢcⱼcⱼᵢ
    end
end

function relax!(particle, velocity, relaxation_probability, transformation_matrix)
    x = particle.position

    rand() > relaxation_probability(x) && return nothing

    u = velocity(x)
    𝓢 = transformation_matrix(x)

    particle.velocity = u + 𝓢 * randn(SVector{3, Float64})
end

function relax!(particle, relax_prob, u, p, T, σ, q, τ, Δt, species)
    x = particle.position

    rand() > relax_prob(x) && return nothing

    particle.velocity = u(x) + sample_usp(p(x), T(x), σ(x), q(x), τ(x), Δt, species)
end

function sample_usp(p, T, σ, q, τ, Δt, species)
    m = species.mass
    Pr = species.prandtl_number

    σ *= 2 / (2 + Δt / τ)
    q *= 2 / (2 + Pr * Δt / τ)

    vₜₕ² = Kᴮ * T / m
    envelope = maximum(abs.(q)) / vₜₕ²^(3/2)
    envelope2 = maximum(abs.(σ[i,j] for (i,j) in ((1,2), (1,3), (2,3)))) / vₜₕ²
    envelope = 1 + 30 * max(envelope, envelope2)

    tao_coth =  Δt\(2τ) * coth(Δt/2τ)
    ψ₁ = (1 - tao_coth) / (2 * vₜₕ² * vₜₕ²)
    ψ₂ = (1 - Pr * tao_coth) / (5 * vₜₕ² * vₜₕ²)

    W = -1
    ξ = zeros(SVector{3, Float64})
    while rand()*envelope > W
        ξ = randn(SVector{3, Float64}) * sqrt(vₜₕ²)
        c² = (ξ' * ξ)
        σᵢⱼc₍ᵢcⱼ₎ = sum((ξ * ξ' - tr(ξ * ξ') / 3 * I) .* σ)
        qkck = (ξ' * q) * (c² / vₜₕ² - 5)

        W = 1 + ψ₁ * σᵢⱼc₍ᵢcⱼ₎ + ψ₂ * qkck
        if W > envelope
            println("W=", W, " envelope=", envelope)
        end
    end

    return ξ
end

most_probable_velocity(T, mₛ) = √(2 * Kᴮ * T / mₛ)

sample_velocity(u, T, mₛ) = u + √0.5 * most_probable_velocity(T, mₛ) * randn()

end