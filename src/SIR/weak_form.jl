# weak_form.jl
# Julia Script

#=
Description: Weak form test for SIR integral elimination
Author: zhangjingyan
Date: 22/06/2026
=#

include("SIRModels.jl")
include("../Measure.jl")
include("../weak_form_helper.jl")

using .Value
using .Logic
using .Integrate
using .Measure
using HomotopyContinuation
using LsqFit
using Random
using Printf
using LinearAlgebra
using QuadGK
using DataInterpolations
using Plots
using BSplineKit
using Statistics


function get_weak_blocks(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method,
    testing_function::Symbol;
    m::Union{Int, Nothing}=nothing,
    order::Union{Int, Nothing}=nothing,
    plot_Ihat::Bool=false
)
    # Since we are differentiating, maybe it is better to write in the original form
    Y  = zeros(K)   # weak left-hand side ∫ phi I'
    W1 = zeros(K)   # ∫ phi I
    W2 = zeros(K)   # ∫ phi I^2
    W3 = zeros(K)   # ∫ phi I F

    t0 = t[1]
    tT = t[end]
    L = tT - t0

    # Using integration by parts:
    # ∫ phi I' = [phi I]_0^T - ∫ phi' I
    # If we ensure [phi I]_0^T = 0, then LHS = - ∫ phi' I

    if method == "S_improved"
        F = Integrate.integrate(t, I_data, "S")

        for k in 1:K
            phi, dphi = get_testing_function(t, k, K, testing_function; m)

            # General weak LHS, including boundary term
            if testing_function == :chebyshev_U
                Y[k] = chebyshev_U_Y_vector(I_data, t, k, method)
            else
                boundary = phi(tT) * I_data[end] - phi(t0) * I_data[1]
                Y[k] = boundary - Integrate.integrate(t, I_data, method; measure=dphi)
            end

            W1[k] = Integrate.integrate(t, I_data, method; measure=phi)
            W2[k] = Integrate.integrate(t, (I_data .^ 2), method; measure=phi)
            W3[k] = Integrate.integrate(t, I_data .* F, method; measure=phi)
        end

    elseif method == "S_formula_improved"
        if testing_function != :sin
            error("S_formula_improved hardcodes sine testing function, invalide testing function choice $testing_function")
        end

        F = Integrate.integrate(t, I_data, "S")

        for k in 1:K
            # General weak LHS, including boundary term
            phi, dphi = Measure.measure_sine_function(t, k)
            boundary = phi(tT) * I_data[end] - phi(t0) * I_data[1]

            Y[k] = boundary - Integrate.integrate(t, I_data, method; t=t, k=k, basis=:sin, derivative=true)

            W1[k] = Integrate.integrate(t, I_data, method; t=t, k=k, basis=:sin)
            W2[k] = Integrate.integrate(t, (I_data .^ 2), method; t=t, k=k, basis=:sin)
            W3[k] = Integrate.integrate(t, I_data .* F, method; t=t, k=k, basis=:sin)
        end

    elseif method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]

        # First approximate I
        # So Ihat is Ihat(x) = c0 + c1 * z + c2 * z^2 + c3 * z^3
        Ihat = build_I_interpolant(t, I_data, method; order=order)

        if plot_Ihat
            t_plot = range(t[1], t[end], length=10000)
            p = plot(
                t_plot,
                Ihat.(t_plot);
                label = "$method interpolant",
                xlabel = "t",
                ylabel = "I(t)"
            )

            scatter!(
                p,
                t,
                I_data;
                label = "Observed data",
                markersize=1,
                markeralpha = 0.3
            )

            display(p)
        end

        # F(x) here we can use DataInterpolations.integrate since we are integrating the interpolation obj from DataInterpolations
        F(x) = DataInterpolations.integral(Ihat, t0, x)

        for k in 1:K
            # General weak LHS, including boundary term
            phi, dphi = get_testing_function(t, k, K, testing_function; m)

            # General weak LHS, including boundary term
            if testing_function == :chebyshev_U
                Y[k] = chebyshev_U_Y(Ihat, t0, tT, k)
            else
                boundary = phi(tT) * I_data[end] - phi(t0) * I_data[1]
                Y[k] = boundary - quadgk(x -> dphi(x) * Ihat(x), t)[1]
            end

            W1[k] = quadgk(x -> phi(x) * Ihat(x), t)[1]
            W2[k] = quadgk(x -> phi(x) * Ihat(x)^2, t)[1]
            W3[k] = quadgk(x -> phi(x) * Ihat(x) * F(x), t)[1]
        end

    elseif method in ["BSpline_GK"]
        println("B_Spline interpolation of order $order")

        # Order 2: straight-line segments
        # Order 3: quadratic segments
        # Order 4: cubic segments
        # Order 6: quintic segments
        Ihat = BSplineKit.interpolate(
            t,
            I_data,
            BSplineOrder(order)
        )
        if plot_Ihat
            p = plot(
                Ihat;
                label = "B-spline interpolant",
                xlabel = "t",
                ylabel = "I(t)"
            )

            scatter!(
                p,
                t,
                I_data;
                label = "Observed data",
                markersize=1,
                markeralpha = 0.3
            )

            display(p)
        end

        Ispline = BSplineKit.Splines.spline(Ihat)

        # Exact analytical antiderivative.
        Fhat = BSplineKit.Splines.integral(Ispline)

        for k in 1:K
            # General weak LHS, including boundary term
            phi, dphi = get_testing_function(t, k, K, testing_function; m)

            # General weak LHS, including boundary term
            if testing_function == :chebyshev_U
                Y[k] = chebyshev_U_Y(Ihat, t0, tT, k)
            else
                boundary = phi(tT) * I_data[end] - phi(t0) * I_data[1]
                Y[k] = boundary - quadgk(x -> dphi(x) * Ihat(x), t)[1]
            end

            W1[k] = quadgk(x -> phi(x) * Ihat(x), t)[1]
            W2[k] = quadgk(x -> phi(x) * Ihat(x)^2, t)[1]
            W3[k] = quadgk(x -> phi(x) * Ihat(x) * Fhat(x), t)[1]
        end

    elseif method in ["QSpline_exact", "CSpline_exact", "Akima_exact"]
        error("not implemented heihei")

    else
        F = Integrate.integrate(t, I_data, method)

        for k in 1:K
            phi, dphi = get_testing_function(t, k, K, testing_function, false; m)

            # General weak LHS, including boundary term
            if testing_function == :chebyshev_U
                Y[k] = chebyshev_U_Y_vector(I_data, t, k, method)
            else
                boundary = phi[end] * I_data[end] - phi[1] * I_data[1]
                Y[k] = boundary - Integrate.integrate(t, dphi .* I_data, method)[end]
            end

            W1[k] = Integrate.integrate(t, phi .* I_data, method)[end]
            W2[k] = Integrate.integrate(t, phi .* (I_data .^ 2), method)[end]
            W3[k] = Integrate.integrate(t, phi .* I_data .* F, method)[end]
        end
    end

    return Y, W1, W2, W3
end


function get_blocks(
    I_data::Vector{Float64},
    t::Vector{Float64},
    method::String
)

    I0 = I_data[1]
    t0 = t[1]

    if !(in_exception(method))
        B1 = Integrate.integrate(t, I_data, method)
        B2 = Integrate.integrate(t, I_data.^2, method)
        B3 = 0.5 .* (B1.^2)

        return I0, B1, B2, B3

    elseif method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]
        Ihat = build_I_interpolant(t, I_data, method)

        B1 = [DataInterpolations.integral(Ihat, t0, x) for x in t]
        B2 = [quadgk(s -> Ihat(s)^2, t0, x)[1] for x in t]
        B3 = 0.5 .* (B1.^2)

        return I0, B1, B2, B3

    else
        error("Unknown integration method $method")
    end
end


function get_W3_IntByParts(I_data::Vector{Float64}, t::Vector{Float64}, K::Int, method)
    W3_parts = zeros(K)

    if method == "S_improved"
        F = Integrate.integrate(t, I_data, "S")

        for k in 1:K
            phi, dphi = Measure.measure_sine_function(t, k)
            boundary = phi(t[end]) * F[end]^2 - phi(t[1]) * F[1]^2
            W3_parts[k] = 0.5 * boundary - 0.5 * Integrate.integrate(t, F.^2, method; measure=dphi)
        end

    elseif method == "S_formula_improved"
        F = Integrate.integrate(t, I_data, "S")

        for k in 1:K
            phi, dphi = Measure.measure_sine_function(t, k)
            boundary = phi(t[end]) * F[end]^2 - phi(t[1]) * F[1]^2
            W3_parts[k] = 0.5 * boundary - 0.5 * Integrate.integrate(t, F.^2, method; t=t, k=k, basis=:sin, derivative=true)
        end

    else
        F = Integrate.integrate(t, I_data, method)

        for k in 1:K
            phi, dphi = Measure.measure_sine(t, k)
            boundary = phi[end] * F[end]^2 - phi[1] * F[1]^2
            W3_parts[k] = 0.5 * boundary - 0.5 * Integrate.integrate(t, dphi .* F.^2, method)[end]
        end
    end

    return W3_parts
end


function _weak_block_analysis(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method_list::Vector{String}
)
    a = t[1]
    b = t[end]
    L = b - a

    for method in method_list
        println()
        println("-------------------------------------")
        println("method = ", method)

        # Original weak blocks
        Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method)
        W3_parts = get_W3_IntByParts(I_data, t, K, method)

        println()
        println("Signed block values, with W3 by parts:")
        @printf("%4s %16s %16s %16s %16s %16s %16s\n",
                "k", "Y[k]", "W1[k]", "W2[k]", "W3 direct", "W3 parts", "diff")

        for k in 1:K
            diff = W3[k] - W3_parts[k]

            @printf("%4d %16.6e %16.6e %16.6e %16.6e %16.6e %16.6e\n",
                    k, Y[k], W1[k], W2[k], W3[k], W3_parts[k], diff)
        end

        println()
        println("Absolute block sizes using W3 by parts:")
        @printf("%4s %16s %16s %16s %16s %16s\n",
                "k", "|Y|", "|W1|", "|W2|", "|W3 parts|", "row norm")

        row_norms = zeros(K)

        for k in 1:K
            row_norms[k] = sqrt(Y[k]^2 + W1[k]^2 + W2[k]^2 + W3_parts[k]^2)

            @printf("%4d %16.6e %16.6e %16.6e %16.6e %16.6e\n",
                    k,
                    abs(Y[k]),
                    abs(W1[k]),
                    abs(W2[k]),
                    abs(W3_parts[k]),
                    row_norms[k])
        end

        println()
        println("Summary using W3 by parts:")

        min_norm = minimum(row_norms)
        max_norm = maximum(row_norms)

        println("min row norm      = ", min_norm)
        println("max row norm      = ", max_norm)

        if min_norm > 0
            println("max/min row norm  = ", max_norm / min_norm)
        else
            println("max/min row norm  = Inf because min row norm is 0")
        end

        A_parts = hcat(W1, W2, W3_parts)

        if K >= 3
            println("condition number of [W1 W2 W3_parts] = ", cond(A_parts))
        else
            println("condition number skipped because K < 3")
        end
    end
end



function weak_block_analysis(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method_list::Vector{String},
    testing_function::Symbol;
    order::Union{Int, Nothing}=nothing
)
    println("======================================")
    println("Weak block analysis")
    println("K = ", K)
    println("Testing function: $testing_function")
    print("Time grid: ", t[1])
    print("-", t[end])
    println(" dt: ", t[2] - t[1])
    println("======================================")
    for method in method_list
        println()
        println("-------------------------------------")
        if method == "BSpline_GK"
            if order === nothing
                error("[weak_block_analysis] BSpline requires order")
            end
        end
        printstyled("method = ", method, color=:yellow, bold=true)
        println()

        Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method, testing_function; order=order, plot_Ihat=false)

        # ------------------------------------------------------------
        # Table 1: actual signed block values
        # ------------------------------------------------------------
        println()
        println("Signed block values:")
        @printf("%4s %16s %16s %16s %16s\n",
                "k", "Y[k]", "W1[k]", "W2[k]", "W3[k]")

        for k in 1:K
            @printf("%4d %16.6e %16.6e %16.6e %16.6e\n",
                    k, Y[k], W1[k], W2[k], W3[k])
        end

        # ------------------------------------------------------------
        # Table 2: row norms and cond value
        # ------------------------------------------------------------
#        println()
#        @printf("%4s %14s %14s %14s %14s %14s\n",
#                "k", "|Y|", "|W1|", "|W2|", "|W3|", "row norm")
#
#        row_norms = zeros(K)
#
#        for k in 1:K
#            row_norms[k] = sqrt(Y[k]^2 + W1[k]^2 + W2[k]^2 + W3[k]^2)
#
#            @printf("%4d %14.4e %14.4e %14.4e %14.4e %14.4e\n",
#                    k,
#                    abs(Y[k]),
#                    abs(W1[k]),
#                    abs(W2[k]),
#                    abs(W3[k]),
#                    row_norms[k])
#        end
#
#        println()
#        println("Summary:")
#        println("min row norm = ", minimum(row_norms))
#        println("max row norm = ", maximum(row_norms))
#        println("max/min row norm = ", maximum(row_norms) / minimum(row_norms))
#
#        A = hcat(W1, W2, W3)
#
#        if K >= 3
#            println("condition number of [W1 W2 W3] = ", cond(A))
#        else
#            println("condition number skipped because K < 3")
#        end
    end
end



function L_hat(paras, I0, W1, W2, W3)
    """
    Vector of length K, containing the RHS of the weak equation.
    When computing residual, we compare Lhat - Y.
    """

    α_eff  = paras[1]
    γ_eff  = paras[2]
    S0_eff = paras[3]

    C1 = (α_eff * (S0_eff + I0) - γ_eff) .* W1
    C2 = -α_eff .* W2
    C3 = -α_eff * γ_eff .* W3

    # I0 still appears through the coefficient α(S0 + I0) - γ
    # It no longer appears as a separate integral block because we differentiated first.
    return C1 .+ C2 .+ C3
end


# TODO: Select best solution based on RSS_Ihat_Idata
# Currently not used!!
function best_solution_weak(solution_list::Vector{Vector{Float64}}, Y::Vector, I0, W1, W2, W3)
    best_sol = Float64[]
    best_err = Inf

    for param in solution_list
        Lhat = L_hat(param, I0, W1, W2, W3)
        err = Logic.get_RSS(Lhat, Y)

        if err <= best_err
            best_err = err
            best_sol = param
        end
    end

    return best_sol, best_err
end


function select_K_weak(
    I_data::Vector{Float64},
    t::Vector{Float64},
    method::String,
    testing_function::Symbol,
    maximum_K::Int,
    threshold::Float64;
    m::Union{Int, Nothing}=nothing,
    order::Union{Int, Nothing}=nothing,
    minimum_K::Int=3,
    consecutive::Int=3,
    if_print::Bool=false
)

    if !(testing_function in [:sin, :chebyshev_U])
        """
        NOTICE: for bump/Hartley/local polynomial functions:
        the first k functions may? change when total K changes
        so this exact cutoff method may be not consistent
        """
        @warn(
            "Block-decay K selection assumes a nested ordered family. " *
            "The selected testing functions may depend on total K."
        )
    end


    Y, W1, W2, W3 = get_weak_blocks(
        I_data,
        t,
        maximum_K,
        method,
        testing_function;
        m=m,
        order=order
    )

    small_count = 0

    for k in 1:maximum_K
        block_size = maximum(abs, (
            Y[k],
            W1[k],
            W2[k],
            W3[k]
        ))

        if block_size <= threshold
            small_count += 1
        else
            small_count = 0
        end

        if small_count >= consecutive
            first_small_k = k - consecutive + 1

            # cut off before the small tail
            K_selected = max(minimum_K, first_small_k - 1)

            if if_print
                println(
                    "Selected K = $K_selected; ",
                    "$consecutive consecutive blocks were below ",
                    "threshold $threshold starting at k = $first_small_k."
                )
            end

            return K_selected
        end
    end

    println("Final block values:")
    println("Y[end]  = ", Y[end])
    println("W1[end] = ", W1[end])
    println("W2[end] = ", W2[end])
    println("W3[end] = ", W3[end])

    @warn(
        "Weak blocks did not remain below threshold $threshold " *
        "for $consecutive consecutive values up to K = $maximum_K."
    )

    return maximum_K
end


function select_T_weak(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method::String,
    testing_function::Symbol;
    m,
    m_min::Int = -6,
    m_max::Int = 6,
    order::Int
)
    Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method, testing_function; m=m, order=order)

    s = [
        norm(Y),
        norm(W1),
        norm(W2),
        norm(W3),
    ]

    powers = [0, 1, 1, 2]

    best_m = nothing
    best_score = Inf

    for m in m_min:m_max
        T = 10.0^m

        scaled = [s[j] / (T^powers[j]) for j in eachindex(s)]

        logs = log10.(scaled .+ eps())
        score = Statistics.var(logs)

        if score < best_score
            best_score = score
            best_m = m
        end
    end

    best_T = 10.0^best_m
    final_scaled = [s[j] / (best_T^powers[j]) for j in eachindex(s)]

    return best_T, final_scaled
end


function select_K_T_weak(
    I_data::Vector{Float64},
    t::Vector{Float64},
    method::String,
    testing_function::Symbol;
    order::Int,
    threshold=1e-2,
    maximum_K=200
)
    """
    alternatively select T and K, since selecting T requires K, but after time rescaling, the block size will change
    """
end



function HC_LS_weak(
    t::Vector{Float64},
    I_data::Vector{Float64},
    vars::Vector,
    method::String,
    testing_function::Symbol;
    K::Union{Int, Nothing}=nothing,
    true_vals=Value.true_vals,
    if_print=true,
    m=10,
    order=3,
    maximum_K::Int=length(t), # N datapoints -> N dimentional vector space
    threshold::Float64=1e-2,
    compare_LS::Bool=false,
    perturb::Float64=0.20, # 20% perturb on true param as initial guess of LS
    LS_iter::Int=5,
    plot_Ihat::Bool=false
)
    """
    YES time rescaling
    YES auto select K based on a threshold on the block size
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """
    if K === nothing
        K = select_K_weak(I_data, t, method, testing_function, maximum_K, threshold; order=order)
    end

    # Since selecting T requires K, if T >= 1, K selected before is still valid for rescaled blocks
    # However if T < 1, blocks can be larger than threshold
    T, _ = select_T_weak(I_data, t, K, method, testing_function; m=m, order=order)

    if T < 1
        @warn(
            "Selected T = $T < 1, so time rescaling enlarges the W blocks " *
            "Rechecking K on the scaled time grid"
        )

        K = select_K_weak(
            I_data,
            t ./ T,
            method,
            testing_function;
            m=m,
            order=order
        )

        T, _ = select_T_weak(
            I_data,
            t,
            K,
            method,
            testing_function;
            m=m,
            order=order
        )
    end

    if T < 1
        # If T < 1 again
        # Though it should not happen that T < 1 ...
        error("method needs update because T always < 1")
    end


    t_scaled = t ./ T
    Y, W1, W2, W3 = get_weak_blocks(I_data, t_scaled, K, method, testing_function; m=m, order=order, plot_Ihat=plot_Ihat)
    B = get_blocks(I_data, t_scaled, method)

    I0 = I_data[1]

    # model for LS, object I_data
    iteration_counts = 0
    function model(x, p)
        iteration_counts += 1
        return Logic.I_hat(p, B...)
    end

    # weak form for HC
    Lhat = L_hat(vars, I0, W1, W2, W3)
    J = sum((Lhat .- Y) .^ 2)

    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results = real_solutions(result)

    if isempty(real_results)
        error("No real HC solution found for SIR weak form.")
    end

    RSS_before = [
        Logic.get_RSS(I_data, Logic.I_hat(r, B...))
        for r in real_results
    ]

    idx_best_before = argmin(RSS_before)
    best_result_beforeLS = real_results[idx_best_before]

    final_results_scaled = Vector{Float64}[]
    RSS_after = Float64[]
    successful_HC_indices = Int[]
    iteration_count_list = Int[]

    lb_scaled = Logic.to_scaled(Value.lb, T)
    ub_scaled = Logic.to_scaled(Value.ub, T)

    for (i, r) in enumerate(real_results)
        p0_row = Float64.(r)

        # Make sure the starting point is inside the LS bounds
        p0 = min.(max.(p0_row, lb_scaled), ub_scaled)

        if p0 != p0_row
            printstyled("Clipping happened. Before clipping: $p0_row; After clipping: $p0\n", color=:red)
        end

        try
            iteration_counts = 0
            fit = curve_fit(
                model,
                t_scaled,
                I_data,
                p0;
                lower = lb_scaled,
                upper = ub_scaled
            )

            push!(final_results_scaled, fit.param)
            push!(RSS_after, Logic.get_RSS(I_data, Logic.I_hat(fit.param, B...)))
            push!(successful_HC_indices, i)
            push!(iteration_count_list, iteration_counts)

        catch e
            @warn "curve_fit failed for initial point" p0 exception=e
        end
    end

    if isempty(final_results_scaled)
        error("No valid LS-refined solutions found.")
    end

    idx_best_after_in_final = argmin(RSS_after)
    idx_best_after_in_HC = successful_HC_indices[idx_best_after_in_final]

    best_result_scaled = final_results_scaled[idx_best_after_in_final]
    best_result = Logic.to_physical(best_result_scaled, T)
    RSS_Ihat_Idata = RSS_after[idx_best_after_in_final]

    if idx_best_before in successful_HC_indices
        pos_before_best_afterLS = findfirst(==(idx_best_before), successful_HC_indices)
        ideal_best_result = final_results_scaled[pos_before_best_afterLS]

        if idx_best_before != idx_best_after_in_HC
            printstyled("Best result before and after LS mismatch\n", color = :red, bold = true)

            println("Best HC index before LS: ", idx_best_before)
            println("Best HC index after LS:  ", idx_best_after_in_HC)

            println("\nBest result before LS:")
            println(best_result_beforeLS)

            println("\nBest-before-LS result after LS:")
            println(ideal_best_result)
            println("RSS after LS from before-best solution: ", RSS_after[pos_before_best_afterLS])

            println("\nBest result after LS:")
            println(best_result_scaled)
            println("RSS after LS from after-best solution: ", RSS_Ihat_Idata)
        end
    else
        printstyled("Warning: the best HC solution before LS failed during LS refinement\n", color = :yellow, bold = true)
        println("Best result before LS: ", best_result_beforeLS)
    end


    if if_print
        printstyled("===== HC_LS_weak SIR Results =====\n", color = :magenta, bold = true)
        printstyled("----------------------------------\n", color = :blue)
        println("Selected K factor: ", K)
        println("Selected time rescale factor: ", T)
        printstyled("Method used: $method", color= :blue)
        println()
        printstyled("Testing function used: $testing_function", color= :blue)
        println()
        println("Number of test functions K: ", K)
        printstyled("----------------------------------\n", color = :blue)

        println("\nBest parameter estimates:")
        for (var, val) in zip(vars, best_result)
            println(var, " = ", val)
        end

        printstyled("----------------------------------\n", color = :blue)
        println("RSS_Ihat_Idata at best weak params = ",
            RSS_Ihat_Idata
        )

        if true_vals !== nothing
            true_scaled = Logic.to_scaled(true_vals, T)

            RSS_Ihat_Idata_true = Logic.get_RSS(I_data, Logic.I_hat(true_scaled, B...))
            println("RSS_Ihat_Idata at true params = ",
                RSS_Ihat_Idata_true
            )

            if RSS_Ihat_Idata < RSS_Ihat_Idata_true
                printstyled("RSS Consistent!\n", color = :red)
            else
                printstyled("RSS NOT Consistent!\n", color = :red)
            end

            parameter_err = Logic.get_param_error(best_result, true_vals)
            println("\nParameter error: ", parameter_err)
        end
        printstyled("----------------------------------\n", color = :blue)

        println("\nALL real results -- #", length(real_results))
        for r in real_results
            printstyled("------------\n", color = :blue)
            r_physical = Logic.to_physical(r, T)
            println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(r, B...)))

            if true_vals !== nothing
                println("parameter error ", Logic.get_param_error(r_physical, true_vals))
            end

            printstyled("------------\n", color = :blue)
        end

        println("\nALL final results -- #", length(final_results_scaled))
        for (ind, r) in enumerate(final_results_scaled)
            printstyled("------------\n", color = :blue)

            if r == best_result_scaled
                printstyled("best result!\n", color=:yellow)
            end
            r_physical = Logic.to_physical(r, T)

            iter_r = iteration_count_list[ind]
            println("model evaluations = ", iter_r)
            println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(r, B...)))

            if true_vals !== nothing
                println("parameter error ", Logic.get_param_error(r_physical, true_vals))
            end
            printstyled("------------\n", color = :blue)
        end

    end


    if compare_LS
        if true_vals === nothing
            error("Comparing with pure LS method requires true vals as reference")
        end

        printstyled("------------ Comparison with LS\n", color = :blue)

        for i in 1:LS_iter
            initial = true_vals .+ perturb .* true_vals .* randn(length(true_vals))

            # Make sure the starting point is inside the LS bounds
            initial = min.(max.(initial, lb_scaled), ub_scaled)

            try
                iteration_counts = 0
                fit = curve_fit(
                    model,
                    t_scaled,
                    I_data,
                    initial;
                    lower = lb_scaled,
                    upper = ub_scaled
                )

                printstyled("------------\n", color = :blue)
                println("$i-th LS with initial guess $initial")
                println("model evaluations = ", iteration_counts)
                println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(fit.param, B...)))
                printstyled("------------\n", color = :blue)
                println()
            catch e
                @warn "curve_fit failed for initial point" initial exception=e
            end
        end
    end

    return best_result, RSS_Ihat_Idata, parameter_err
end

















function _HC_LS_weak(
    t::Vector{Float64},
    I_data::Vector{Float64},
    vars::Vector,
    method::String,
    testing_function::Symbol;
    K::Int = 8,
    true_vals=Value.true_vals,
    if_print=true,
    m=10,
    order=3
)
    """
    YES time rescaling
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """
    T, _ = select_T_weak(I_data, t, K, method, testing_function; m=m, order=order)
    t_scaled = t ./ T
    Y, W1, W2, W3 = get_weak_blocks(I_data, t_scaled, K, method, testing_function; m=m, order=order)

    I0 = I_data[1]

    function model(x, p)
        return L_hat(p, I0, W1, W2, W3)
    end

    Lhat = L_hat(vars, I0, W1, W2, W3)
    J = sum((Lhat .- Y) .^ 2)

    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results = real_solutions(result)

    if isempty(real_results)
        error("No real HC solution found for SIR weak form.")
    end

    RSS_before = [
        Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3))
        for r in real_results
    ]

    idx_best_before = argmin(RSS_before)
    best_result_beforeLS = real_results[idx_best_before]

    final_results_scaled = Vector{Float64}[]
    RSS_after = Float64[]
    successful_HC_indices = Int[]

    xdata = collect(1:K)
    lb_scaled = Logic.to_scaled(Value.lb, T)
    ub_scaled = Logic.to_scaled(Value.ub, T)

    for (i, r) in enumerate(real_results)
        p0 = Float64.(r)

        # Make sure the starting point is inside the LS bounds
        p0 = min.(max.(p0, lb_scaled), ub_scaled)

        try
            fit = curve_fit(
                model,
                xdata,
                Y,
                p0;
                lower = lb_scaled,
                upper = ub_scaled
            )

            push!(final_results_scaled, fit.param)
            push!(RSS_after, Logic.get_RSS(Y, L_hat(fit.param, I0, W1, W2, W3)))
            push!(successful_HC_indices, i)

        catch e
            @warn "curve_fit failed for initial point" p0 exception=e
        end
    end

    if isempty(final_results_scaled)
        error("No valid LS-refined solutions found.")
    end

    idx_best_after_in_final = argmin(RSS_after)
    idx_best_after_in_HC = successful_HC_indices[idx_best_after_in_final]

    best_result_scaled = final_results_scaled[idx_best_after_in_final]
    best_result = Logic.to_physical(best_result_scaled, T)
    RSS = RSS_after[idx_best_after_in_final]

    if idx_best_before in successful_HC_indices
        pos_before_best_afterLS = findfirst(==(idx_best_before), successful_HC_indices)
        ideal_best_result = final_results_scaled[pos_before_best_afterLS]

        if idx_best_before != idx_best_after_in_HC
            printstyled("Best result before and after LS mismatch\n", color = :red, bold = true)

            println("Best HC index before LS: ", idx_best_before)
            println("Best HC index after LS:  ", idx_best_after_in_HC)

            println("\nBest result before LS:")
            println(best_result_beforeLS)

            println("\nBest-before-LS result after LS:")
            println(ideal_best_result)
            println("RSS after LS from before-best solution: ", RSS_after[pos_before_best_afterLS])

            println("\nBest result after LS:")
            println(best_result_scaled)
            println("RSS after LS from after-best solution: ", RSS)
        end
    else
        printstyled("Warning: the best HC solution before LS failed during LS refinement\n", color = :yellow, bold = true)
        println("Best result before LS: ", best_result_beforeLS)
    end

    parameter_err = Logic.get_param_error(best_result, true_vals)

    if in_exception(method)
        B = Logic.get_blocks(I_data, t_scaled, "S")
    else
        B = Logic.get_blocks(I_data, t_scaled, method)
    end

    Ihat_best = Logic.I_hat(best_result_scaled, B...)

    RSS_Ihat_Idata = Logic.get_RSS(Ihat_best, I_data)

    if if_print
        printstyled("===== HC_LS_weak SIR Results =====\n", color = :magenta, bold = true)
        println("Selected time rescale factor: ", T)
        printstyled("Method used: $method", color= :blue)
        println()
        printstyled("Testing function used: $testing_function", color= :blue)
        println()
        println("Number of test functions K: ", K)

        println("\nBest parameter estimates:")
        for (var, val) in zip(vars, best_result)
            println(var, " = ", val)
        end

        println("\nResidual sum of squares (RSS_Lhat_L(Y)): ", RSS)
        println("Residual sum of squares (RSS_Ihat_Idata): ", RSS_Ihat_Idata)

        true_scaled = Logic.to_scaled(true_vals, T)

        println("RSS weak at true params = ",
            Logic.get_RSS(Y, L_hat(true_scaled, I0, W1, W2, W3))
        )

        println("RSS weak at best params = ",
            Logic.get_RSS(Y, L_hat(best_result_scaled, I0, W1, W2, W3))
        )

        println("pointwise RSS at true params = ",
            Logic.get_RSS(I_data, Logic.I_hat(true_scaled, B...))
        )

        println("pointwise RSS at best weak params = ",
            Logic.get_RSS(I_data, Logic.I_hat(best_result_scaled, B...))
        )


        println("\nParameter error: ", parameter_err)

        println("ALL real results -- #", length(real_results))
        for r in real_results
            r_physical = Logic.to_physical(r, T)
            println("RSS ", Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3)))
            println("parameter error ", Logic.get_param_error(r_physical, true_vals))
        end

        println("ALL final results -- #", length(final_results_scaled))
        for r in final_results_scaled
            if r == best_result_scaled
                printstyled("best result!\n", color=:yellow)
            end
            r_physical = Logic.to_physical(r, T)

            println("RSS ", Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3)))
            println("parameter error ", Logic.get_param_error(r_physical, true_vals))
        end

    end

    return best_result, RSS, RSS_Ihat_Idata, parameter_err
end


"""
NOT UPDATED
This version uses NO time rescaling
"""
function __HC_LS_weak(
    t::Vector{Float64},
    I_data::Vector{Float64},
    vars::Vector,
    method::String;
    K::Int = 8,
    true_vals=Value.true_vals,
    if_print=true
)
    """
    No time rescaling
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """

    Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method)

    I0 = I_data[1]

    function model(x, p)
        return L_hat(p, I0, W1, W2, W3)
    end

    Lhat = L_hat(vars, I0, W1, W2, W3)
    J = sum((Lhat .- Y) .^ 2)

    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results = real_solutions(result)

    if isempty(real_results)
        error("No real HC solution found for SIR weak form.")
    end

    RSS_before = [
        Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3))
        for r in real_results
    ]

    idx_best_before = argmin(RSS_before)
    best_result_beforeLS = real_results[idx_best_before]

    final_results = Vector{Float64}[]
    RSS_after = Float64[]
    successful_HC_indices = Int[]

    xdata = collect(1:K)

    for (i, r) in enumerate(real_results)
        p0 = Float64.(r)

        # Make sure the starting point is inside the LS bounds
        p0 = min.(max.(p0, Value.lb), Value.ub)

        try
            fit = curve_fit(
                model,
                xdata,
                Y,
                p0;
                lower = Value.lb,
                upper = Value.ub
            )

            push!(final_results, fit.param)
            push!(RSS_after, Logic.get_RSS(Y, L_hat(fit.param, I0, W1, W2, W3)))
            push!(successful_HC_indices, i)

        catch e
            @warn "curve_fit failed for initial point" p0 exception=e
        end
    end

    if isempty(final_results)
        error("No valid LS-refined solutions found.")
    end

    idx_best_after_in_final = argmin(RSS_after)
    idx_best_after_in_HC = successful_HC_indices[idx_best_after_in_final]

    best_result = final_results[idx_best_after_in_final]
    RSS = RSS_after[idx_best_after_in_final]

    if idx_best_before in successful_HC_indices
        pos_before_best_afterLS = findfirst(==(idx_best_before), successful_HC_indices)
        ideal_best_result = final_results[pos_before_best_afterLS]

        if idx_best_before != idx_best_after_in_HC
            printstyled("Best result before and after LS mismatch\n", color = :red, bold = true)

            println("Best HC index before LS: ", idx_best_before)
            println("Best HC index after LS:  ", idx_best_after_in_HC)

            println("\nBest result before LS:")
            println(best_result_beforeLS)

            println("\nBest-before-LS result after LS:")
            println(ideal_best_result)
            println("RSS after LS from before-best solution: ", RSS_after[pos_before_best_afterLS])

            println("\nBest result after LS:")
            println(best_result)
            println("RSS after LS from after-best solution: ", RSS)
        end
    else
        printstyled("Warning: the best HC solution before LS failed during LS refinement\n", color = :yellow, bold = true)
        println("Best result before LS: ", best_result_beforeLS)
    end

    parameter_err = Logic.get_param_error(best_result, true_vals)

    if if_print
        if in_exception(method)
            B = Logic.get_blocks(I_data, t, "S")
        else
            B = Logic.get_blocks(I_data, t, method)
        end
        Ihat_best = Logic.I_hat(best_result, B...)

        printstyled("===== HC_LS_weak SIR Results =====\n", color = :magenta, bold = true)
        println("No time rescaling")
        printstyled("Method used: $method", color= :blue)
        println()
        println("Number of test functions K: ", K)

        println("\nBest parameter estimates:")
        for (var, val) in zip(vars, best_result)
            println(var, " = ", val)
        end

        println("\nResidual sum of squares (RSS_Lhat_L(Y)): ", RSS)
        println("Residual sum of squares (RSS_Ihat_Idata): ", Logic.get_RSS(Ihat_best, I_data))

        println("RSS weak at true params = ",
            Logic.get_RSS(Y, L_hat(true_vals, I0, W1, W2, W3))
        )

        println("RSS weak at best params = ",
            Logic.get_RSS(Y, L_hat(best_result, I0, W1, W2, W3))
        )

        println("pointwise RSS at true params = ",
            Logic.get_RSS(I_data, Logic.I_hat(true_vals, B...))
        )

        println("pointwise RSS at best weak params = ",
            Logic.get_RSS(I_data, Logic.I_hat(best_result, B...))
        )

        println("\nParameter error: ", parameter_err)

        println("ALL real results -- #", length(real_results))
        for r in real_results
            println("RSS ", Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3)))
            println("parameter error ", Logic.get_param_error(r, true_vals))
        end

        println("ALL final results -- #", length(final_results))
        for r in final_results
            if r == best_result
                printstyled("best result!\n", color=:yellow)
            end

            println("RSS ", Logic.get_RSS(Y, L_hat(r, I0, W1, W2, W3)))
            println("parameter error ", Logic.get_param_error(r, true_vals))
        end

    end

    return best_result, RSS, parameter_err
end
