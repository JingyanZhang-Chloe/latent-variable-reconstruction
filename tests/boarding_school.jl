# boarding_school.jl
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
    S0_true_ratio = (763 - 1) / 763   # if starting from Jan 21
    true_vals = Float64[α_true_ratio, γ_true, S0_true_ratio]


    # Boarding school influenza data
    days = Float64[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    I_data_row = Float64[3, 25, 75, 227, 296, 258, 236, 192, 126, 71, 28, 11, 7]
    I_data = I_data_row ./ N
    t = days .- first(days)

    results = Logic.HC_LS(t, I_data, variables, "S", true_vals=true_vals)
    Logic.print_HC_LS(results)

    # Weak-form version
    HC_LS_weak(t, I_data, variables, "S"; K=6, true_vals=true_vals)
    println()

    # NOTICE parameter error here has no meaning!!!! since it is refer to the data we manually generate
end

main()
