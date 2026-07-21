# Eyam plague.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/07/2026

(1) The Eyam plague
See Section 4 in https://arxiv.org/pdf/1603.03819
=#


include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic
using HomotopyContinuation

@var α, γ, S0
const variables = [α, γ, S0]



function main()
    N = 350.0

    t = Float64[
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0
    ]

    S_data_row = Float64[
        254,
        235,
        201,
        153,
        121,
        110,
        97,
        83
    ]

    I_data_row = Float64[
        7,
        14,
        22,
        29,
        20,
        8,
        8,
        0
    ]

    S_data = S_data_row ./ N
    I_data = I_data_row ./ N

    # The author uses a simple approximation method for the forward diﬀerential equation and
    # comes up with a point estimate (3.39,0.0212)
    true_vals = Float64[
        0.0212 * N,
        3.39,
        254 / N
    ]

    """
    Other vals in the paper to compare with:

    bayesian = [6.895, 3.22, 254 / 350]
    deterministic = [6.23, 2.73, 254 / 350]
    raggett = [7.42, 3.39, 254 / 350]
    """

    results = Logic.HC_LS(t, I_data, variables, "S", true_vals=true_vals)
    Logic.print_HC_LS(results)

    HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U; true_vals=true_vals, threshold=0.2, compare_LS=true)

    println()
end

main()
