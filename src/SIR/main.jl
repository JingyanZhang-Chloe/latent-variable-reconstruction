# main.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 25/06/2026
=#

include("SIRModels.jl")
include("weak_form.jl")
using .Value
using .Logic

@var α, γ, S0
const variables = [α, γ, S0]


function main()

    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)

    results = Logic.HC_LS(t, I_data, variables, "S")
    Logic.print_HC_LS(results)

    HC_LS_weak(t, I_data, variables, "S_improved"; K=10)
    HC_LS_weak(t, I_data, variables, "S"; K=10)
    println()
end

main()

