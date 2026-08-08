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
    n_control_points::Union{Int, Nothing}=nothing,
    d_smooth::Union{Int, Nothing}=nothing,
    λ::Union{Int, Nothing}=nothing,
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

    elseif method in ["QSpline_GK", "CSpline_GK", "Akima_GK", "BSpline_GK", "BSplineApprox", "RegularizationSmooth", "PCHIP"]

        if method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]
            # First approximate I
            # So Ihat is Ihat(x) = c0 + c1 * z + c2 * z^2 + c3 * z^3
            Ihat = build_I_interpolant(t, I_data, method; order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)

            # F(x) here we can use DataInterpolations.integrate since we are integrating the interpolation obj from DataInterpolations
            Fhat(x) = DataInterpolations.integral(Ihat, t0, x)

        elseif method in ["BSpline_GK"]
            # Order 2: straight-line segments
            # Order 3: quadratic segments
            # Order 4: cubic segments
            # Order 6: quintic segments
            Ihat = BSplineKit.interpolate(
                t,
                I_data,
                BSplineOrder(order)
            )

            Ispline = BSplineKit.Splines.spline(Ihat)

            # Exact analytical antiderivative.
            Fhat = BSplineKit.Splines.integral(Ispline)

        elseif method in ["BSplineApprox", "RegularizationSmooth", "PCHIP"]

            Ihat = build_I_interpolant(t, I_data, method; order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)

            F_values = cumulative_quadgk(Ihat, t)

            # TODO: Check how to interpolate F_values properly
            Fhat = DataInterpolations.CubicSpline(
                F_values,
                t
            )
        end


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

    return Y, W1, W2, W3
end


function get_blocks(
    I_data::Vector{Float64},
    t::Vector{Float64},
    method::String;
    order::Union{Int, Nothing}=nothing,
    n_control_points::Union{Int, Nothing}=nothing,
    d_smooth::Union{Int, Nothing}=nothing,
    λ::Union{Int, Nothing}=nothing
)

    I0 = I_data[1]
    t0 = t[1]

    if !(in_exception(method))
        B1 = Integrate.integrate(t, I_data, method)
        B2 = Integrate.integrate(t, I_data.^2, method)
        B3 = 0.5 .* (B1.^2)

        return I0, B1, B2, B3

    elseif method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]
        Ihat = build_I_interpolant(t, I_data, method, order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)

        B1 = [DataInterpolations.integral(Ihat, t0, x) for x in t]
        B2 = cumulative_quadgk(s -> Ihat(s)^2, t)
        B3 = 0.5 .* (B1.^2)

        return I0, B1, B2, B3

    elseif method in ["BSplineApprox", "RegularizationSmooth", "PCHIP"]
        Ihat = build_I_interpolant(t, I_data, method, order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)

        B1 = cumulative_quadgk(Ihat, t)
        B2 = cumulative_quadgk(s -> Ihat(s)^2, t)
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
    n_control_points::Union{Int, Nothing}=nothing,
    d_smooth::Union{Int, Nothing}=nothing,
    λ::Union{Int, Nothing}=nothing,
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
        order=order,
        n_control_points=n_control_points,
        d_smooth=d_smooth,
        λ=λ
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
    order::Union{Int, Nothing}=nothing,
    n_control_points::Union{Int, Nothing}=nothing,
    d_smooth::Union{Int, Nothing}=nothing,
    λ::Union{Int, Nothing}=nothing
)

    Y, W1, W2, W3 = get_weak_blocks(
        I_data,
        t,
        K,
        method,
        testing_function;
        m=m,
        order=order,
        n_control_points=n_control_points,
        d_smooth=d_smooth,
        λ=λ
    )

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
    n_control_points=max(5, round(Int, 0.3 * length(t))),
    d_smooth::Union{Int, Nothing}=2,
    λ::Union{Int, Nothing}=100,
    maximum_K::Int=length(t), # N datapoints -> N dimentional vector space
    threshold::Float64=1e-2,
    print_root_record::Bool=true,
    compare_LS::Bool=false,
    perturb::Float64=0.20, # 20% perturb on true param as initial guess of LS
    LS_iter::Int=5,
    plot_Ihat::Bool=false,
    profiling::Bool=false
)
    """
    YES time rescaling
    YES auto select K based on a threshold on the block size
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """

    times = Dict{String, Float64}()

    time_start = time()
    if K === nothing
        K = select_K_weak(I_data, t, method, testing_function, maximum_K, threshold; order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)
    end

    # Since selecting T requires K, if T >= 1, K selected before is still valid for rescaled blocks
    # However if T < 1, blocks can be larger than threshold
    T, _ = select_T_weak(I_data, t, K, method, testing_function; m=m, order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)

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
            order=order,
            n_control_points=n_control_points,
            d_smooth=d_smooth,
            λ=λ
        )

        T, _ = select_T_weak(
            I_data,
            t,
            K,
            method,
            testing_function;
            m=m,
            order=order,
            n_control_points=n_control_points,
            d_smooth=d_smooth,
            λ=λ
        )
    end

    if T < 1
        # If T < 1 again
        # Though it should not happen that T < 1 ...
        error("method needs update because T always < 1")
    end

    times["select K T"] = time() - time_start

    time_start = time()
    t_scaled = t ./ T

    Y, W1, W2, W3 = get_weak_blocks(
        I_data,
        t,
        K,
        method,
        testing_function;
        m=m,
        order=order,
        n_control_points=n_control_points,
        d_smooth=d_smooth,
        λ=λ,
        plot_Ihat=plot_Ihat
    )

    times["get weak blocks"] = time() - time_start

    time_start = time()
    B = get_blocks(I_data, t_scaled, method, order=order, n_control_points=n_control_points, d_smooth=d_smooth, λ=λ)
    times["get blocks"] = time() - time_start

    time_start = time()
    I0 = I_data[1]

    iteration_counts = 0
    function model(x, p)
        iteration_counts += 1
        return Logic.I_hat(p, B...)
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
    times["HC"] = time() - time_start

    lb_scaled = Logic.to_scaled(Value.lb, T)
    ub_scaled = Logic.to_scaled(Value.ub, T)

    records = RootRecord[]

    final_results_scaled = Union{Vector{Float64}, Nothing}[]
    final_results_scaled_empty = true

    hc_rss_list = Float64[]
    ls_rss_list = Float64[]

    time_start = time()
    for (i, r) in enumerate(real_results)
        hc_root = Float64.(r)
        hc_rss = Logic.get_RSS(I_data, Logic.I_hat(r, B...))
        push!(hc_rss_list, hc_rss)

        # Make sure the starting point is inside the LS bounds
        clipped_root = min.(max.(hc_root, lb_scaled), ub_scaled)
        clipped_rss = Logic.get_RSS(I_data, Logic.I_hat(clipped_root, B...))

        clipped = clipped_root != hc_root

        try
            iteration_counts = 0
            fit = curve_fit(
                model,
                t_scaled,
                I_data,
                clipped_root;
                lower = lb_scaled,
                upper = ub_scaled
            )

            ls_root = fit.param
            ls_rss = Logic.get_RSS(I_data, Logic.I_hat(ls_root, B...))

            record = RootRecord(
                index = i,
                hc_root = hc_root,
                hc_rss = hc_rss,

                clipped = clipped,
                clipped_root = clipped_root,
                clipped_rss = clipped_rss,

                ls_success = true,
                ls_root = ls_root,
                ls_rss = ls_rss,
                ls_iterations = iteration_counts,

                error_message = nothing
            )

            push!(final_results_scaled, ls_root)
            push!(ls_rss_list, ls_rss)
            push!(records, record)

            final_results_scaled_empty = false

        catch e
            record = RootRecord(
                index = i,
                hc_root = hc_root,
                hc_rss = hc_rss,

                clipped = clipped,
                clipped_root = clipped_root,
                clipped_rss = clipped_rss,

                ls_success = false,
                ls_root = nothing,
                ls_rss = nothing,
                ls_iterations = nothing,

                error_message = e
            )

            push!(final_results_scaled, nothing)
            push!(ls_rss_list, Inf)
            push!(records, record)
        end
    end

    if final_results_scaled_empty
        error("No valid LS-refined solutions found.")
    end
    times["LS"] = time() - time_start

    hc_best_index = argmin(hc_rss_list)
    ls_best_index = argmin(ls_rss_list)

    if hc_best_index != ls_best_index
        @warn "Best result before and after LS mismatch \n HC best index: $hc_best_index \n LS best index: $ls_best_index"
    end

    best_result_scaled = final_results_scaled[ls_best_index]
    best_result = Logic.to_physical(best_result_scaled, T)
    RSS_Ihat_Idata = ls_rss_list[ls_best_index]

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
    end


    if print_root_record
        printstyled("------------ Root records\n", color = :blue)

        for record in records
            print_root_info(record)
        end
    end

    if compare_LS
        if true_vals === nothing
            error("Comparing with pure LS method requires true vals as reference")
        end

        printstyled("------------ Comparison with LS\n", color = :blue)
        println("perturb: ", perturb)

        for i in 1:LS_iter
            initial = true_vals .+ perturb .* true_vals .* randn(length(true_vals))

            # Make sure the starting point is inside the LS bounds
            initial = min.(max.(initial, lb_scaled), ub_scaled)
            initial = Logic.to_scaled(initial, T)

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
                println("final answer: ", Logic.to_physical(fit.param, T))
                printstyled("------------\n", color = :blue)
                println()
            catch e
                @warn "curve_fit failed for initial point" initial exception=e
            end
        end
    end

    if profiling
        printstyled("------------ Timing\n", color = :blue)
        for (name, t) in times
            println("$name : $(round(t, digits=4)) seconds")
        end
    end

    return best_result, RSS_Ihat_Idata, parameter_err
end