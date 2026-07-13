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
#    t = collect(0.0:10.0:1000.0)
#
#    S, I, R = Logic.simulate_sir(t)
#
#    noise = 0
#    I_data = I .+ noise .* I .* randn(length(I))
#
#    # Optional: avoid negative infected values after adding noise
#    I_data = max.(I_data, 0.0)
#
#    weak_block_analysis(I_data, t, 30, ["BSpline_GK", "CSpline_GK"], :chebyshev_U; degree=4)

    for dt in [10.0, 5.0, 2.0, 1.0, 0.5]
        t = collect(0.0:dt:1000.0)
        S, I, R = Logic.simulate_sir(t)

        weak_block_analysis(
            I,
            t,
            30,
            ["BSpline_GK", "CSpline_GK"],
            :chebyshev_U,
        )
    end


    for tT in [100.0, 200.0, 500.0, 1000.0]
        t = collect(0.0:1.0:tT)
        S, I, R = Logic.simulate_sir(t)

        weak_block_analysis(
            I,
            t,
            30,
            ["BSpline_GK", "CSpline_GK"],
            :chebyshev_U,
        )
    end
end

main()
