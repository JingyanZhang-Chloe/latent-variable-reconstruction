# Tristan da Cunha island.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/07/2026

(2) Common cold outbreak data in Tristan da Cunha island (1967)
See Table 3 in https://arxiv.org/pdf/2208.12113
=#


include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation

@var α, γ, S0
const variables = [α, γ, S0]



function main()
    t = collect(0.0:20.0)

    I_data_row = Float64[
        1, 1, 3, 7, 6, 10, 13,
        13, 14, 14, 17, 10, 6, 6,
        4, 3, 1, 1, 1, 1, 0
    ]

    R_data_row = Float64[
        0, 0, 0, 0, 5, 7, 8,
        13, 13, 16, 16, 24, 30, 31,
        33, 34, 36, 36, 36, 36, 37
    ]

    # the number of susceptible individuals is not directly observed and thus S(0) needs to be estimated
    # So we don't know the total count N :(
    # Following the setting in Toni et al.(2009)
    # S0 ≈ 38 based on the Approximated posterior densities under the SIR model

    N = 39.0

    I_data = I_data_row ./ N

    # PINTS implementation of the Tristan da Cunha ordinary SIR model????
    # pints.toy.SIRModel using the Tristan da Cunha common-cold data
    true_vals = Float64[
        1.014,
        0.285,
        38 / 39
    ]

    try
        results = Logic.HC_LS(t, I_data, variables, "S", true_vals=true_vals)
        Logic.print_HC_LS(results)
    catch e
        println("HC_LS failed. Error message $e")
    end

    # HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U; true_vals=true_vals, threshold=0.5, compare_LS=true, plot_Ihat=true)
    HC_LS_weak(t, I_data, variables, "BSplineApprox", :chebyshev_U; true_vals=true_vals, threshold=0.5, compare_LS=true, plot_Ihat=true)

    println()
end

main()
