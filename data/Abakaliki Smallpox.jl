# Abakaliki Smallpox.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/07/2026

(3) Abakaliki Smallpox
See Table 1 in https://arxiv.org/pdf/1605.07924
=#

#=
the table records rash-onset times for individual cases r

By the paper table 3 Durations of disease stages in the smallpox model: the approximated infectious interval is
[r - 2.49, r + 16]

So we need to count, for each day, how many residents are in that infectious inverval
=#


include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation

@var α, γ, S0
const variables = [α, γ, S0]


function main()
    # rash_days[i] is the date of onset of rash for case i
    rash_days = Int[
        0, 13, 20, 22, 25, 25, 25, 26,
        30, 35, 28, 40, 40, 42, 42, 47,
        50, 51, 55, 55, 56, 56, 57, 58,
        60, 60, 61, 63, 66, 66, 71, 76
    ]

    # We have in total 76 days
    t = collect(0.0:76.0)

    # For each day, how many rash happened
    onset_incidence = Float64[
        count(==(day), rash_days)
        for day in 0:76
    ]

    # from table 3
    μ_fever_to_rash = 2.49
    μ_rash_to_removal = 16.0

    # Each day, how many residence are in infectious mode
    I_data = Float64[
        count(
            r -> r - μ_fever_to_rash <= day < r + μ_rash_to_removal,
            rash_days
        )
        for day in t
    ]

    results = Logic.HC_LS(t, I_data, variables, "S", true_vals=nothing)
    Logic.print_HC_LS(results)

    HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U; true_vals=nothing, threshold=0.2)
end

main()
