# RSS_analysis.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 15/07/2026
=#

include("../src/SIR/SIRModels.jl")
include("../src/SIR/weak_form.jl")
using .Value
using .Logic

@var α, γ, S0
const variables = [α, γ, S0]



using Statistics
using Plots
using Printf

function RSS_analysis(
    t::Vector{Float64},
    I::Vector{Float64},
    noise::Float64,
    vars::Vector,
    method::String,
    testing_function::Symbol,
    K_list::Vector{Int};
    trials::Int = 50
)
    mean_RSS = zeros(length(K_list))
    std_RSS  = zeros(length(K_list))

    println("RSS_Ihat_Idata stability analysis")
    println("---------------------------------------------------------------")
    @printf("%5s %14s %14s %10s %14s %14s\n",
            "K", "Mean RSS/K", "Std RSS/K", "CV", "Minimum", "Maximum")
    println("---------------------------------------------------------------")

    for (j, K) in enumerate(K_list)
        RSS_trials = zeros(trials)

        for trial in 1:trials
            I_data = I .+ noise .* I .* randn(length(I))

            _, _, RSS, _ = HC_LS_weak(
                t,
                I_data,
                vars,
                method,
                testing_function;
                K = K,
                if_print = false
            )

            RSS_trials[trial] = RSS / K
        end

        mean_RSS[j] = mean(RSS_trials)
        std_RSS[j]  = std(RSS_trials)

        cv = std_RSS[j] / mean_RSS[j]

        @printf(
            "%5d %14.6e %14.6e %10.4f %14.6e %14.6e\n",
            K,
            mean_RSS[j],
            std_RSS[j],
            cv,
            minimum(RSS_trials),
            maximum(RSS_trials)
        )
    end

    println("---------------------------------------------------------------")

    p = plot(
        K_list,
        mean_RSS;
        # ribbon = std_RSS,
        marker = :circle,
        xlabel = "Number of test functions K",
        ylabel = "Normalized RSS (RSS / K)",
        title = "RSS_Ihat_Idata stability across noisy trials",
        label = "Mean ± 1 standard deviation"
    )

    return p
end


function main()
    testing_function = :chebyshev_U
    method = "CSpline_GK"
    noise = 0.05

    t = collect(0.0:10.0:1000.0)
    S, I, R = Logic.simulate_sir(t)

    RSS_analysis(
        t,
        I,
        noise,
        variables,
        method,
        testing_function,
        collect(10:10:100)
    )
end

main()
