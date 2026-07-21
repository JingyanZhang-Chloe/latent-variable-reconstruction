# Boarding School.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 25/06/2026
=#

include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation

@var α, γ, S0
const variables = [α, γ, S0]


function main()
    # Some true parameters
    N = 763.0

    α_true_ratio = 0.002342 * N
    γ_true = 0.476

    # Boarding school influenza data
    # January 21 is t = 0, with one infected student
    days = Float64[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    I_data_row = Float64[
        1, 3, 25, 75, 227, 296, 258,
        236, 192, 126, 71, 28, 11, 7
    ]

    I_data = I_data_row ./ N
    t = days



    S0_true_ratio = (763 - 1) / 763   # if starting from Jan 21
    true_vals = Float64[α_true_ratio, γ_true, S0_true_ratio]


    results = Logic.HC_LS(t, I_data, variables, "S", true_vals=true_vals)
    Logic.print_HC_LS(results)

    # Weak-form version
    # K=4
#    HC_LS_weak(t, I_data, variables, "S", :chebyshev_U; true_vals=true_vals)
#    HC_LS_weak(t, I_data, variables, "S_improved", :chebyshev_U; true_vals=true_vals)
    HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U; true_vals=true_vals, threshold=0.2, compare_LS=true)
    HC_LS_weak(t, I_data, variables, "BSpline_GK", :chebyshev_U; order=4, true_vals=true_vals, threshold=0.2, compare_LS=true)
    println()
end

main()
