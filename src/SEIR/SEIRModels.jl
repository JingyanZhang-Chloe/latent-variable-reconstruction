# SEIRModels.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 18/06/2026
=#

include("../Integrate.jl")

module Value
    const S0 = 0.99
    const E0 = 0.0
    const I0 = 0.01
    const R0 = 0.0

    const α = 0.2
    const σ = 0.01
    const γ = 0.005

    const p_true = [α, σ, γ]
    const u = [S0, E0, I0, R0]
    const true_vals = [α, σ, γ, S0, E0]
    const scales = [0.01, 0.01, 0.01, 1.0, 1.0]

    const lb = [0.0, 0.0, 0.0, 0.8, 0.0] # Forcing S0 to be greater than 0.8
    const ub = [Inf, Inf, Inf, 1.0, 0.2]

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


    function seir!(du, u, p, t)
        S, E, I, R = u
        α, σ, γ = p
        du[1] = - α * S * I
        du[2] = α * S * I - σ * E
        du[3] = σ * E - γ * I
        du[4] = γ * I
    end


    function simulate_seir(t; u0=Value.u, p=Value.p_true, plot=false)
        prob = ODEProblem(seir!, u0, (t[1], t[end]), p)
        sol = DifferentialEquations.solve(prob, saveat=t)
        sol_arr = Array(sol)
        S = sol_arr[1, :]
        E = sol_arr[2, :]
        I = sol_arr[3, :]
        R = sol_arr[4, :]

        if plot
            data_to_plot = hcat(S, E, I, R)
            println("Plotting data of size: ", size(data_to_plot))
            plt = Plots.plot(t, data_to_plot,
            title = "SEIR Model Results",
            label = ["True S" "True E" "True I" "True R"],
            xlabel = "Time",
            ylabel = "Value",
            lw = 2
            )
            display(plt)
        end

        return S, E, I, R
    end


    function get_blocks(I_data::Vector{Float64}, t::Vector{Float64}, method::String)
        I0 = I_data[1]

        if method == "T"
            I_int = cumul_integrate(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = cumul_integrate(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- cumul_integrate(t, t .* I_data)
            B5 = t .* cumul_integrate(t, I_data.^2) .- cumul_integrate(t, t .* (I_data.^2))
            B6 = cumul_integrate(t, (I_int).^2)
        elseif method == "S"
            I_int = Integrate.cumintegrate(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = Integrate.cumintegrate(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- Integrate.cumintegrate(t, t .* I_data)
            B5 = t .* Integrate.cumintegrate(t, I_data.^2) .- Integrate.cumintegrate(t, t .* (I_data.^2))
            B6 = Integrate.cumintegrate(t, (I_int).^2)
        elseif method == "S_uniform"
            I_int = Integrate.cumintegrate_simpson_uniform(t, I_data)
            B1 = 1
            B2 = I_int
            B3 = Integrate.cumintegrate_simpson_uniform(t, I_data.^2 .- I0^2)
            B4 = t .* I_int .- Integrate.cumintegrate_simpson_uniform(t, t .* I_data)
            B5 = t .* Integrate.cumintegrate_simpson_uniform(t, I_data.^2) .- Integrate.cumintegrate_simpson_uniform(t, t .* (I_data.^2))
            B6 = Integrate.cumintegrate_simpson_uniform(t, (I_int).^2)
        else
            @error "method must be T, S, or S_uniform"
        end

        return I0, B1, B2, B3, B4, B5, B6
    end


    function I_hat(paras, I0, B1, B2, B3, B4, B5, B6, t; scale=false)
        if scale
            α_eff  = paras[1] * Value.scales[1]
            σ_eff  = paras[2] * Value.scales[2]
            γ_eff  = paras[3] * Value.scales[3]
            S0_eff = paras[4] * Value.scales[4]
            E0_eff = paras[5] * Value.scales[5]
        else
            α_eff  = paras[1]
            σ_eff  = paras[2]
            γ_eff  = paras[3]
            S0_eff = paras[4]
            E0_eff = paras[5]
        end

        C1 = σ_eff * (E0_eff + I0) .* t .* B1
        C2 = - (γ_eff + σ_eff) .* B2
        C3 = - 0.5 * α_eff .* B3
        C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* B4
        C5 = - α_eff * (γ_eff + σ_eff) .* B5
        C6 = - 0.5 * α_eff * σ_eff * γ_eff .* B6

        return I0 .+ C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
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
        sigma_list = Float64[]
        gamma_list = Float64[]
        S0_list = Float64[]
        E0_list = Float64[]

        blocks = get_blocks(I_data, t, method)

        function model(x, p)
            push!(alpha_list, p[1])
            push!(sigma_list, p[2])
            push!(gamma_list, p[3])
            push!(S0_list, p[4])
            push!(E0_list, p[5])

            return I_hat(p, blocks..., x)
        end

        fit = curve_fit(model, t, I_data, u0, lower=Value.lb, upper=Value.ub)
        p_hat = fit.param

        return (
            α_trace = alpha_list,
            σ_trace = sigma_list,
            γ_trace = gamma_list,
            S0_trace = S0_list,
            E0_trace = E0_list,
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


    function print_results(results::NamedTuple)
        true_list = results.true_params
        estimated_list = results.estimated
        initial_list = results.initial_guesses
        t = results.t
        err_list = get_param_error(estimated_list, true_list)

        cost = sum((true_list .- estimated_list).^2)
        iteration = length(results.α_trace)

        rss = nothing
        if results.I !== nothing
            Ihat = I_hat(estimated_list, results.blocks..., t)
            rss = get_RSS(Ihat, results.I) # RSS (sum((I_hat .- I_data).^2))
        end

        println("=" ^ 82)
        println("Estimation Results")
        println("Total cost: $cost | iteration steps: $iteration | RSS (sum((I_hat .- I_data).^2)): $rss")
        println("=" ^ 82)

        param_labels = String["α", "σ", "γ", "S0", "E0"]

        for (label, t_val, i_val, e_val, err_val) in zip(param_labels, true_list, initial_list, estimated_list, err_list)
            println("----- $label -----")
            println("True: $t_val | Guess: $i_val | Result: $e_val | Error: $err_val")
        end
    end


    function plot_results(results::NamedTuple)
        p1 = plot(results.α_trace, title="Alpha Convergence", color=:orange, label="Est Alpha", m=:o, ms=3)
        hline!([results.true_params[1]], label="True", color=:black, ls=:dash)

        p2 = plot(results.σ_trace, title="Sigma Convergence", color=:purple, label="Est Sigma", m=:o, ms=3)
        hline!([results.true_params[2]], label="True", color=:black, ls=:dash)

        p3 = plot(results.γ_trace, title="Gamma Convergence", color=:green, label="Est Gamma", m=:o, ms=3)
        hline!([results.true_params[3]], label="True", color=:black, ls=:dash)

        p4 = plot(results.S0_trace, title="S0 Convergence", color=:blue, label="Est S0", m=:o, ms=3)
        hline!([results.true_params[4]], label="True", color=:black, ls=:dash)

        p5 = plot(results.E0_trace, title="E0 Convergence", color=:red, label="Est E0", m=:o, ms=3)
        hline!([results.true_params[5]], label="True", color=:black, ls=:dash)

        final_plot = plot(p1, p2, p3, p4, p5, layout=(1, 5), size=(1600, 300))

        display(final_plot)
    end


    function best_solution(solution_list::Vector{Vector{Float64}}, I_data::Vector, I0, B1, B2, B3, B4, B5, B6, t::Vector)
        best_sol = Float64[]
        best_err = Inf

        for param in solution_list
            Ihat = I_hat(param, I0, B1, B2, B3, B4, B5, B6, t)
            err = get_RSS(Ihat, I_data)
            if err <= best_err
                best_err = err
                best_sol = param
            end
        end

        return best_sol, best_err
    end


    function select_T(I_data, t; method="S", m_min=-6, m_max=6)
        I0, B1, B2, B3, B4, B5, B6 = get_blocks(I_data, t, method)

        s = [
            B2[end],  # scales as 1/T
            B3[end],  # scales as 1/T
            B4[end],  # scales as 1/T^2
            B5[end],  # scales as 1/T^2
            B6[end],  # scales as 1/T^3
        ]
        powers = [1, 1, 2, 2, 3]   # exponent of T in denominator

        best_m = nothing
        best_score = Inf

        for m in m_min:m_max
            T = 10.0^m  # use float to allow negative m cleanly
            scaled = [ s[j] / (T^powers[j]) for j in eachindex(s) ]
            logs = log10.(scaled .+ eps())  # eps() avoids log(0)
            score = var(logs)

            if score < best_score
                best_score = score
                best_m = m
            end
        end

        best_T = 10.0^best_m
        final_scaled = [ s[j] / (best_T^powers[j]) for j in eachindex(s) ]

        return best_T, final_scaled
    end


    to_physical(res_scaled, T::Float64) = [res_scaled[1] / T, res_scaled[2] / T, res_scaled[3] / T, res_scaled[4], res_scaled[5]]
    to_scaled(res, T::Float64) = [res[1] * T, res[2] * T, res[3] * T, res[4], res[5]]


    function swap_project_SE0_for_sigma_gamma!(x::Vector{Float64}, I0::Float64)
        α, σ, γ, S0, E0 = x
        r = σ / γ

        σ_new = γ
        γ_new = σ

        # preserve σ(E0+I0)
        E0_new = r * (E0 + I0) - I0

        # preserve σ(S0+E0+I0)
        S0_new = r * S0

        x[2] = σ_new
        x[3] = γ_new
        x[4] = S0_new
        x[5] = E0_new
        return x
    end


    function project_S0E0_euclidean!(x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})
        s0 = x[4]
        e0 = x[5]

        if s0 + e0 <= 1.0
            x[4] = s0
            x[5] = e0
            return x
        end

        # Feasible s must satisfy:
        #   lbS <= s <= ubS
        #   lbE <= 1-s <= ubE  =>  1-ubE <= s <= 1-lbE
        s_min = max(lb[4], 1.0 - ub[5])
        s_max = min(ub[4], 1.0 - lb[5])

        # Euclidean projection (smallest adjustment in L2)
        s_star = 0.5 * (s0 + 1.0 - e0)
        s_proj = clamp(s_star, s_min, s_max)
        e_proj = 1.0 - s_proj

        x[4] = s_proj
        x[5] = e_proj
        return x
    end


    function project_to_bounds(result::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, I0::Float64)::Vector{Float64}
        """
        Here bounds we are applying is
        all(lb_scaled .<= res .<= ub_scaled) && (res[2] > res[3]) && (res[4] + res[5] <= 1)
        """
        x = copy(result)

        if x[2] < x[3]
            swap_project_SE0_for_sigma_gamma!(x, I0)
        end

        x = clamp.(x, lb, ub)

        if lb[4] + lb[5] > 1
            @warn "Infeasible bounds: lb[4] + lb[5] > 1, cannot satisfy S0 + E0 <= 1"
            return x
        end

        if x[4] + x[5] > 1
            project_S0E0_euclidean!(x, lb, ub)
        end

        x = clamp.(x, lb, ub)

        return x
    end


    function HC_LS(t::Vector{Float64}, I_data::Vector{Float64}, vars::Vector, method::String; I=nothing, true_vals=Value.true_vals)
        T, _ = select_T(I_data, t)
        t_scaled = t ./ T
        B = get_blocks(I_data, t_scaled, method)

        function model(x, p)
            return I_hat(p, B..., x)
        end

        Ihat = I_hat(vars, B..., t_scaled)
        J = sum((Ihat .- I_data).^2)
        system_eqs = differentiate(J, vars)
        C = System(system_eqs, variables=vars)
        result = HomotopyContinuation.solve(C, show_progress=false)
        real_results_scaled = real_solutions(result)

        lb_scaled = to_scaled(Value.lb, T)
        ub_scaled = to_scaled(Value.ub, T)

        final_results_scaled = Vector{Float64}[]

        for r in real_results_scaled
            bound_r = project_to_bounds(r, lb_scaled, ub_scaled, B[1])
            fit = curve_fit(model, t_scaled, I_data, bound_r, lower=lb_scaled, upper=ub_scaled)
            push!(final_results_scaled, fit.param)
        end

        best_result_scaled, RSS_Ihat_Idata = best_solution(final_results_scaled, I_data, B..., t_scaled)
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

        println("=== HC_LS SEIR Results ===")
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
