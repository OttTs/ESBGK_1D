include("../ESBGK_1D3V.jl")

using .ESBGK_1D3V
using GLMakie: lines, Observable, notify, lines!
using LinearAlgebra:diag

# Initial state
n = 1.37E19
T = 273
u = [0, 0, 0]

# Init species
m = 6.63E-26
ω = 1E14
Pr = 2/3
Tᵣ = 273
ωᵣ = 0.77
dᵣ = 4.05E-10
species = ESBGK_1D3V.Species(m, ω, Pr, Tᵣ, ωᵣ; ref_diameter=dᵣ)

# Setup mesh
BCₗ = ESBGK_1D3V.DiffuseWallBC([0, 500, 0], T)
BCᵣ = ESBGK_1D3V.DiffuseWallBC([0, -500, 0], T)
mesh = ESBGK_1D3V.Mesh(10, (-0.5, 0.5))

# time step...
Δt = 1E-5

# Linear or Constant solution
function_space = ESBGK_1D3V.Lagrange(mesh)                      # Linear
#function_space = ESBGK_1D3V.DiscontinuousLagrangeLagrange(mesh) # Constant

## ----------------------------------------------------------------------------------------
# Sample initial data and simulate for 0.25s
particles = ESBGK_1D3V.init_uniform(n, u, T, ESBGK_1D3V.limits(mesh), m, species.weighting)

t = 0.25
ESBGK_1D3V.simulate!(particles, species, Δt, t, mesh, (BCₗ, BCᵣ); function_space)

## ----------------------------------------------------------------------------------------
# Sample and plot current temperature
∑v⁰, ∑v¹, ∑v² = ESBGK_1D3V.sample_moments(particles, function_space)

velocity = ESBGK_1D3V.with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
    return ∑v¹ᵢ / ∑v⁰ᵢ
end

temperature = ESBGK_1D3V.with(∑v⁰, ∑v², velocity) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
    ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
    return species.mass / (3 * ESBGK_1D3V.Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
end

x = -0.499:0.01:0.499
y = Observable(temperature.(x))
lines!(ax, x, y)
fig,ax = lines(x, y) # Only use this for the first plot!

## ----------------------------------------------------------------------------------------
# Run for a long time and plot the solution after each time step
# Use to see if the solution is "converged" or to average the result
tend = 1
i = 0
while t < tend
    i += 1
    t += Δt
    ESBGK_1D3V.simulate!(particles, species, Δt, Δt, mesh, (BCₗ, BCᵣ); function_space)

    ∑v⁰, ∑v¹, ∑v² = ESBGK_1D3V.sample_moments(particles, function_space)

    velocity = ESBGK_1D3V.with(∑v¹, ∑v⁰) do ∑v¹ᵢ, ∑v⁰ᵢ
        return ∑v¹ᵢ / ∑v⁰ᵢ
    end

    using LinearAlgebra:diag
    temperature = ESBGK_1D3V.with(∑v⁰, ∑v², velocity) do ∑v⁰ᵢ, ∑v²ᵢ, uᵢ
        ∑c² = sum(diag(∑v²ᵢ)) - ∑v⁰ᵢ * uᵢ' * uᵢ
        return species.mass / (3 * ESBGK_1D3V.Kᴮ) * ∑c² / (∑v⁰ᵢ - 1)
    end

    y[] = temperature.(x)                       # Uncomment to display current state only
    #y[] = (y[]*i + temperature.(x)) / (i + 1) # Uncomment for averaging
    yield()
end