# main.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/06/2026
=#

include("SEIRModels.jl")
include("weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation
using Random
using Statistics
using Printf

@var αT, σT, γT, S0, E0
const variables = [αT, σT, γT, S0, E0]

function main()
    t = collect(0.0:10.0:1000.0)
    S, E, I, R = Logic.simulate_seir(t)
    noise = 0.01
    println("Noise level: $noise")
    I_data = I .+ noise .* I .* randn(length(I))
    I_data = max.(I_data, 0.0)

    results = Logic.HC_LS(t, I_data, variables, "S")
    Logic.print_HC_LS(results)
#
    # HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U, compare_LS=true)
    println()

#    t_scaled = t ./ 100
#    method = "CSpline_GK"
#    B = get_blocks(I_data, t_scaled, method)

#    a1 = Logic.get_RSS(I_data, Logic.I_hat(
#        [20.001161706971413, 0.5000024522749338, 0.9999844291864545, 1.0, 0.009890269518319193],
#        B...,
#        t_scaled
#    ))
#
#    a2 = Logic.get_RSS(I_data, Logic.I_hat(
#        [0,0,0,1,0.2],
#        B...,
#        t_scaled
#    ))
#
#    println("$a1   $a2")
end

main()
