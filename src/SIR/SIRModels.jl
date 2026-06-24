# SIRModels.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 22/06/2026
=#

include("../Integrate.jl")

module Value
    const S0 = 0.99
    const I0 = 0.01
    const R0 = 0.0

    const α = 0.2
    const γ = 0.005

    const p_true = [α, γ]
    const u = [S0, I0, R0]
    const true_vals = [α, γ, S0]

    const lb = [0.0, 0.0, 0.8] # Forcing S0 to be greater than 0.8
    const ub = [Inf, Inf, 1.0]

    const T = 100.0
end


module Logic
    using ..Value
    using ..Integrate
    using DifferentialEquations
    using Plots
    using NumericalIntegration
    using HomotopyContinuation
    using Statistics
    using LsqFit
    using Random


    function sir!(du, u, p, t)
        S, I, R = u
        α, γ = p
        du[1] = - α * S * I
        du[2] = α * S * I - γ * I
        du[3] = γ * I
    end


    function simulate_sir(t; u0=Value.u, p=Value.p_true, plot=false)
        prob = ODEProblem(sir!, u0, (t[1], t[end]), p)
        sol = DifferentialEquations.solve(prob, saveat=t)
        sol_arr = Array(sol)
        S = sol_arr[1, :]
        I = sol_arr[2, :]
        R = sol_arr[3, :]

        if plot
            data_to_plot = hcat(S, I, R)
            println("Plotting data of size: ", size(data_to_plot))
            plt = Plots.plot(t, data_to_plot,
            title = "SIR Model Results",
            label = ["True S" "True I" "True R"],
            xlabel = "Time",
            ylabel = "Value",
            lw = 2
            )
            display(plt)
        end

        return S, I, R
    end


    function get_blocks(I_data::Vector{Float64}, t::Vector{Float64}, method::String)
        I0 = I_data[1]

        B1 = Integrate.integrate(t, I_data, method)
        B2 = Integrate.integrate(t, I_data.^2, method)
        B3 = 0.5 .* (B1.^2)

        return I0, B1, B2, B3
    end


    function I_hat(paras, I0, B1, B2, B3; scale=false)
        α_eff = paras[1]
        γ_eff = paras[2]
        S0_eff = paras[3]

        C1 = (α_eff * (S0_eff + I0) - γ_eff) .* B1
        C2 = - α_eff .* B2
        C3 = - α_eff * γ_eff .* B3

        return I0 .+ C1 .+ C2 .+ C3
    end


    function least_squares(
        u0::Vector,
        I_data::Vector,
        t::Vector,
        method::String;
        I::Union{Nothing, Vector{Float64}}=nothing
    )
        """
        Perform least squres method with no rescale
        """

        alpha_list = Float64[]
        gamma_list = Float64[]
        S0_list = Float64[]

        blocks = get_blocks(I_data, t, method)

        function model(x, p)
            push!(alpha_list, p[1])
            push!(gamma_list, p[2])
            push!(S0_list, p[3])

            return I_hat(p, blocks...)
        end

        fit = curve_fit(model, t, I_data, u0, lower=Value.lb, upper=Value.ub)
        p_hat = fit.param

        return (
            α_trace = alpha_list,
            γ_trace = gamma_list,
            S0_trace = S0_list,
            estimated = p_hat,
            true_params = [Value.α, Value.σ, Value.γ, Value.S0, Value.E0],
            t = t,
            initial_guesses = u0,
            I_data = I_data,
            I = I,
            blocks = blocks
        )
    end


    function get_param_error(est::Vector, true_vals::Vector=Value.true_vals)::Vector
        return abs.(est .- true_vals) ./ true_vals .* 100
    end


    function get_RSS(est::Vector, true_value::Vector)::Float64
        return sum((est .- true_value).^2)
    end


    function best_solution(solution_list::Vector{Vector{Float64}}, I_data::Vector, I0, B1, B2, B3)
        best_sol = Float64[]
        best_err = Inf

        for param in solution_list
            Ihat = I_hat(param, I0, B1, B2, B3)
            err = get_RSS(Ihat, I_data)
            if err <= best_err
                best_err = err
                best_sol = param
            end
        end

        return best_sol, best_err
    end


    function select_T(I_data, t; method="S", m_min=-6, m_max=6)
        I0, B1, B2, B3 = get_blocks(I_data, t, method)

        s = [
            B1[end],  # ∫ I, scales as 1/T
            B2[end],  # ∫ I^2, scales as 1/T
            B3[end],  # 1/2 (∫ I)^2, scales as 1/T^2
        ]

        powers = [1, 1, 2]   # exponent of T in denominator

        best_m = nothing
        best_score = Inf

        for m in m_min:m_max
            T = 10.0^m
            scaled = [s[j] / (T^powers[j]) for j in eachindex(s)]
            logs = log10.(scaled .+ eps())
            score = var(logs)

            if score < best_score
                best_score = score
                best_m = m
            end
        end

        best_T = 10.0^best_m
        final_scaled = [s[j] / (best_T^powers[j]) for j in eachindex(s)]

        return best_T, final_scaled
    end


    to_physical(res_scaled, T::Float64) = [res_scaled[1] / T, res_scaled[2] / T, res_scaled[3]]
    to_scaled(res, T::Float64) = [res[1] * T, res[2] * T, res[3]]


    function HC_LS(t::Vector{Float64}, I_data::Vector{Float64}, vars::Vector, method::String; I=nothing, true_vals=Value.true_vals)
        T, _ = select_T(I_data, t)
        t_scaled = t ./ T
        B = get_blocks(I_data, t_scaled, method)

        function model(x, p)
            return I_hat(p, B...)
        end

        Ihat = I_hat(vars, B...)
        J = sum((Ihat .- I_data).^2)
        system_eqs = differentiate(J, vars)
        C = System(system_eqs, variables=vars)
        result = HomotopyContinuation.solve(C, show_progress=false)
        real_results_scaled = real_solutions(result)

        lb_scaled = to_scaled(Value.lb, T)
        ub_scaled = to_scaled(Value.ub, T)

        final_results_scaled = Vector{Float64}[]

        for r in real_results_scaled
            bound_r = min.(max.(r, lb_scaled), ub_scaled)
            fit = curve_fit(model, t_scaled, I_data, bound_r, lower=lb_scaled, upper=ub_scaled)
            push!(final_results_scaled, fit.param)
        end

        if isempty(final_results_scaled)
            error("No real HC solution found for SIR.")
        end

        best_result_scaled, RSS_Ihat_Idata = best_solution(final_results_scaled, I_data, B...)
        best_result = to_physical(best_result_scaled, T)
        parameter_err = get_param_error(best_result, true_vals)
        if I != nothing
            RSS_Idata_I = get_RSS(I_data, I)
        else
            RSS_Idata_I = nothing
        end

        return (
            method = method,
            best_result = best_result,
            parameter_err = parameter_err,
            RSS_Ihat_Idata = RSS_Ihat_Idata,
            RSS_Idata_I = RSS_Idata_I,
            vars=vars
        )
    end


    function print_HC_LS(results::NamedTuple)

        println("=== HC_LS SIR Results ===")
        println("Method used: ", results.method)

        println("\nBest parameter estimates:")
        for (var, val) in zip(results.vars, results.best_result)
            println(var, " = ", val)
        end

        println("\nResidual sum of squares (RSS_Ihat_Idata): ", results.RSS_Ihat_Idata)

        println("\nParameter error: ", results.parameter_err)

        if results.RSS_Idata_I != nothing
            println("\nBaseline residual sum of squares (RSS_Idata_I): ", results.RSS_Idata_I)
        end
    end

end
