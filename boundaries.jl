struct DiffuseWallBC
    velocity::SVector{3, Float64}
    temperature::Float64
end

function collide!(particle, bc; species)
    particle.velocity = sample_inflow_velocity(bc.velocity, bc.temperature, species.mass)
end

function sample_inflow_velocity(u, T, mₛ)
    uₘₚ = most_probable_velocity(T, mₛ)
    zs = samplezs(u[1] / uₘₚ)
    return SVector(
        u[1] - zs * uₘₚ,
        u[2] + √0.5 * uₘₚ * randn(),
        u[3] + √0.5 * uₘₚ * randn()
    )
end

function samplezs(a::Number)
    # Samples the random variable zs with a given speed ratio a
    # See "Garcia and Wagner - 2006 - Generation of the Maxwellian inflow distribution"
    if a < -0.4
        z = 0.5*(a - √(a^2+2))
        β = a - (1 - a) * (a - z)
        while true
            if exp(-β^2) / (exp(-β^2) + 2 * (a - z) * (a - β) * exp(-z^2)) > rand()
                zs = -√(β^2 - log(rand()))
                (zs - a) / zs > rand() && return zs
            else
                zs = β + (a - β) * rand()
                (a - zs) / (a - z) * exp(z^2 - zs^2) > rand() && return zs
            end
        end
    elseif a < 0
        while true
            zs = -√(a^2 - log(rand()))
            (zs - a) / zs > rand() && return zs
        end
    elseif a < 1.3
        while true
            u = rand()
            a * √π / (a * √π + 1 + a^2) > u && return -1/√2 * abs(randn())
            (a * √π + 1) / (a * √π + 1 + a^2) > u && return -√(-log(rand()))
            zs = (1 - √rand()) * a
            exp(-zs^2) > rand() && return zs
        end
    else # a > 1.3
        while true
            if 1 / (2 * a * √π + 1) > rand()
                zs = -√(-log(rand()))
            else
                zs = 1 / √2 * randn()
            end
            (a - zs) / a > rand() && return zs
        end
    end
end