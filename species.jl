struct Species
    mass::Float64
    weighting::Float64
    prandtl_number::Float64
    ref_temperature::Float64
    ref_viscosity::Float64
    ref_exponent::Float64
end

function Species(mass, weighting, prandtl_number, ref_temperature, ref_exponent; ref_diameter)
    μᵣ = reference_viscosity(mass, ref_temperature, ref_diameter, ref_exponent)
    return Species(mass, weighting, prandtl_number, ref_temperature, μᵣ, ref_exponent)
end

reference_viscosity(mₛ, Tᵣ, dᵣ, ω) = 30 * √(mₛ * Kᴮ * Tᵣ / π) / (4 * (5 - 2 * ω) * (7 - 2 * ω) * dᵣ^2)
