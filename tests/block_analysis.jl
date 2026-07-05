# block_analysis.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/07/2026
=#

include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation

@var α, γ, S0
const variables = [α, γ, S0]

function main()
    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)

    weak_block_analysis(I_data, t, 12, ["S", "S_improved", "S_formula_improved"])
end

main()
