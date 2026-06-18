# homotopy_continuation.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/06/2026
=#

include("SEIRModels.jl")
using .Value
using .Logic
using HomotopyContinuation

@var αT, σT, γT, S0, E0
const variables = [αT, σT, γT, S0, E0]

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    results = Logic.HC_LS(t, I_data, variables, "S")
    Logic.print_HC_LS(results)
end

main()
