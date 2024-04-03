include("../ESBGK_1D3V.jl")

using .ESBGK_1D3V
using GLMakie: lines, Observable, notify, lines!
using LinearAlgebra:diag

# Initial state
n = 1.37E21
T = 273
u = [0, 0, 0]

# Init species
m = 6.63E-26
ω = 1E16
Pr = 2/3
Tᵣ = 273
ωᵣ = 0.77
dᵣ = 4.05E-10
species = ESBGK_1D3V.Species(m, ω, Pr, Tᵣ, ωᵣ; ref_diameter=dᵣ)

# Setup mesh
BCₗ = ESBGK_1D3V.DiffuseWallBC([0, 500, 0], T)
BCᵣ = ESBGK_1D3V.DiffuseWallBC([0, -500, 0], T)
mesh = ESBGK_1D3V.Mesh(25, (-0.5, 0.5))
conservemesh = ESBGK_1D3V.Mesh(25, (-0.5, 0.5))
# time step...
Δt = 16*6.25E-7

# Linear or Constant solution
function_space = ESBGK_1D3V.Lagrange{1}(mesh)              # Linear
function_space = ESBGK_1D3V.DiscontinuousLagrange{1}(mesh) # Constant
function_space = ESBGK_1D3V.StochasticFunction(mesh)

## ----------------------------------------------------------------------------------------
# Sample initial data and simulate for 0.25s
particles = ESBGK_1D3V.init_uniform(n, u, 273, ESBGK_1D3V.limits(mesh), m, species.weighting)

t = 0.25
ESBGK_1D3V.simulate!(particles, species, Δt, t, mesh, (BCₗ, BCᵣ); function_space)#, do_USP=true)

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

x = -0.4999:0.0001:0.4999
y = Observable(∑v⁰.(x))
y = Observable(temperature.(x))
lines!(ax, x, y)
fig,ax = lines(x, y) # Only use this for the first plot!

tau = ESBGK_1D3V.with(∑v⁰, temperature) do ∑v⁰ᵢ, Tᵢ
    n = species.weighting / ESBGK_1D3V.element_size(mesh) * ∑v⁰ᵢ
    p = n * ESBGK_1D3V.Kᴮ * Tᵢ
    μ = ESBGK_1D3V.dynamic_viscosity(Tᵢ, species.ref_viscosity, species.ref_temperature, species.ref_exponent)
    τ = 1 / species.prandtl_number * μ / p
    return τ
end

## ----------------------------------------------------------------------------------------
# Run for a long time and plot the solution after each time step
# Use to see if the solution is "converged" or to average the result
tend = 1
i = 0
t=0
while t < tend
    i += 1
    t += Δt
    ESBGK_1D3V.simulate!(particles, species, Δt, Δt, mesh, (BCₗ, BCᵣ); function_space, conservation_mesh=conservemesh)#, do_USP=true)

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
    #y[] = ∑v⁰.(x) * ESBGK_1D3V.num_elements(mesh)                   # Uncomment to display current state only
    #y[] = (y[]*i + ∑v⁰.(x)* ESBGK_1D3V.num_elements(mesh) ) / (i + 1) # Uncomment for averaging
    yield()
end


##
using CSV, DataFrames
data = CSV.read("examples/kn-0_001-dvm.csv", DataFrame)
using GLMakie:plot!
plot!(ax, data.Points_0, data.DVM_Temperature)