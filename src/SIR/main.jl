# main.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 25/06/2026
=#

include("SIRModels.jl")
include("weak_form.jl")
using .Value
using .Logic

@var α, γ, S0
const variables = [α, γ, S0]


using Random
using Statistics
using Printf

function one_trial(trial; K=8, noise=0.01)
    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    I_data = I .+ noise .* I .* randn(length(I))
    I_data = max.(I_data, 0.0)

    res_standard = Logic.HC_LS(t, I_data, variables, "S")

    res_weak_improved = HC_LS_weak(t, I_data, variables, "S_improved"; K=K, if_print=false)
    res_weak_standard = HC_LS_weak(t, I_data, variables, "S"; K=K, if_print=false)

    return (
        trial = trial,
        standard = res_standard,
        weak_improved = res_weak_improved,
        weak_standard = res_weak_standard,
    )
end


function stability_test(; n_trials=20, K=8, noise=0.01)
    all_results = []

    for trial in 1:n_trials
        try
            result = one_trial(trial; K=K, noise=noise)
            push!(all_results, result)

        catch e
            println("Trial $trial failed:")
            println(e)
        end

    end

    return all_results
end


function main()

    # t = collect(0.0:1.0:100.0)
    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    noise = 0.01
    println("Noise level: $noise")
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)

#    results = Logic.HC_LS(t, I_data, variables, "S")
#    Logic.print_HC_LS(results)
#
     # K = 60
#    # before time rescaling
#    _HC_LS_weak(t, I_data, variables, "S_improved"; K=K)
#    # after time rescaling
#
#    HC_LS_weak(t, I_data, variables, "S_improved"; K=K)
#    println()


#    HC_LS_weak(t, I_data, variables, "S_improved", :sin; K=K)
#    HC_LS_weak(t, I_data, variables, "S_improved", :bump; K=K)
#    HC_LS_weak(t, I_data, variables, "S_improved", :hartley; K=K)
#    HC_LS_weak(t, I_data, variables, "S_improved", :polynomial; K=K)
#    HC_LS_weak(t, I_data, variables, "S_improved", :chebyshev_U; K=K)

    HC_LS_weak(t, I_data, variables, "BSpline_GK", :chebyshev_U; order=6)
    HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U)

#    for degree in 3:8
#        HC_LS_weak(t, I_data, variables, "BSpline_GK", :sin; K=K, degree=degree)
#        HC_LS_weak(t, I_data, variables, "BSpline_GK", :chebyshev_U; K=K, degree=degree)
#    end

#    _HC_LS_weak(t, I_data, variables, "S_improved"; K=K)

#    HC_LS_weak(t, I_data, variables, "QSpline_GK"; K=K)
#    _HC_LS_weak(t, I_data, variables, "QSpline_GK"; K=K)
#    # results = stability_test(n_trials=20, K=8, noise=0.01)
#
#    HC_LS_weak(t, I_data, variables, "CSpline_GK"; K=K)
#    _HC_LS_weak(t, I_data, variables, "CSpline_GK"; K=K)
#
#    HC_LS_weak(t, I_data, variables, "Akima_GK"; K=K)
#    _HC_LS_weak(t, I_data, variables, "Akima_GK"; K=K)

    println()
end

main()

