# 2 Zellen, periodische Ränder
# Insgesamt num_samples samples
# Relaxationswahrscheinlichkeit: 0.5
# Zellgröße 1, Zeitschritt 1


num_samples = 10^2
num_steps = 10^3

samples = randn(num_samples)
variances = Float64[]
means = Float64[]
for i in 1:num_steps
    μ = mean(samples)
    σ² = var(samples)#; corrected=false)

    samples = randn(num_samples)*√σ² .+ μ

    push!(means, μ)
    push!(variances, σ²)
end
println("μ=", round(mean(samples); digits=4), ", σ²=", round(var(samples); digits=4))

using GLMakie

lines(means)
lines!(variances)


