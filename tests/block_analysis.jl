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
using BSplineKit

@var α, γ, S0
const variables = [α, γ, S0]


"""
Compute

    ∫_{t0}^{tT} [T_n(x(t)) / sqrt(1 - x(t)^2)] * g(t) dt

using x = cos(θ)
"""
function chebyshev_T_integral(
    g,
    t0::Real,
    tT::Real,
    k::Int,
)
    L = tT - t0
    n = k - 1

    t_from_theta(θ) =
        (t0 + tT) / 2 + (L / 2) * cos(θ)

    value, _ = quadgk(
        θ -> cos(n * θ) * g(t_from_theta(θ)),
        0.0,
        π,
    )

    return (L / 2) * value
end


function chebyshev_T_test_W1(t, I_data, K)
    Ihat = BSplineKit.interpolate(
        t,
        I_data,
        BSplineOrder(4),
    )

    K = 60
    W1_T = zeros(K)

    for k in 1:K
        W1_T[k] = chebyshev_T_integral(
            Ihat,
            t[1],
            t[end],
            k,
        )
    end

    println("Chebyshev-T W1:")
    for k in 1:K
        println("k = $k, n = $(k - 1), W1 = $(W1_T[k])")
    end
end


function main()
    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    noise = 0
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)


    weak_block_analysis(I_data, t, 100, ["BSpline_GK"], :chebyshev_U; order=4)

#    for dt in [10.0, 5.0, 2.0, 1.0, 0.5]
#        t = collect(0.0:dt:1000.0)
#        S, I, R = Logic.simulate_sir(t)
#
#        weak_block_analysis(
#            I,
#            t,
#            30,
#            ["CSpline_GK"],
#            :chebyshev_U,
#            degree=4
#        )
#    end


#    for tT in [100.0, 200.0, 500.0, 1000.0]
#        t = collect(0.0:1.0:tT)
#        S, I, R = Logic.simulate_sir(t)
#
#        weak_block_analysis(
#            I,
#            t,
#            30,
#            ["CSpline_GK"],
#            :chebyshev_U,
#            degree=4
#        )
#    end
end

main()
