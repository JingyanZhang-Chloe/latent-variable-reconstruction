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

    t = collect(0.0:10.0:1000.0)

    S, I, R = Logic.simulate_sir(t)

    noise = 0.01
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)

    results = Logic.HC_LS(t, I_data, variables, "S")
    Logic.print_HC_LS(results)

    K = 12
    HC_LS_weak(t, I_data, variables, "S_improved"; K=K)
    HC_LS_weak(t, I_data, variables, "S"; K=K)
    println()

    # results = stability_test(n_trials=20, K=8, noise=0.01)
end

main()

