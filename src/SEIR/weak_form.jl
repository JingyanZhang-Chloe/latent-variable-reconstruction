# weak_form.jl
# Julia Script

#=
Description: Weak form test for SEIR integral elimination
Author: zhangjingyan
Date: 19/06/2026
=#


include("SEIRModels.jl")
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


function get_weak_blocks(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method::String,
    testing_function::Symbol;
    m::Union{Int, Nothing}=nothing,
    order::Union{Int, Nothing}=nothing,
    plot_Ihat::Bool=false
)

    I0 = I_data[1]

    # Since we are differentiating, maybe it is better to write in the orginal form
    Y  = zeros(K)   # weak left-hand side ∫ phi I'
    W1 = zeros(K)   # ∫ phi
    W2 = zeros(K)   # ∫ phi I
    W3 = zeros(K)   # ∫ phi (I^2 - I0^2)
    W4 = zeros(K)   # ∫ phi F
    W5 = zeros(K)   # ∫ phi G
    W6 = zeros(K)   # ∫ phi F^2
    # Using integration by parts: ∫ phi I' = [phi I]_0^T - ∫ phi' I
    # If we ensure [phi I]_0^T = 0 we have LHS: - ∫ phi' I

    t0 = t[1]
    tT = t[end]
    L = tT - t0

    if method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]

        Ihat = build_I_interpolant(t, I_data, method; order=order)

        if plot_Ihat
            p = plot(
                Ihat;
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

        # F_values = Integrate.integrate(t, I_data, "S")
        # G_values = Integrate.integrate(t, I_data.^2, "S")

        # Issue: hang for a very long time
        # F(x) = DataInterpolations.integral(Ihat, t0, x)
        # G(x) = quadgk(s -> Ihat(s)^2, t0, x)[1]

        F_values = Float64[]
        F_current = 0.0

        G_values = Float64[]
        G_current = 0.0

        for i in eachindex(t)
            if i == firstindex(t)
                push!(F_values, F_current)
                push!(G_values, G_current)
            else
                F_current += DataInterpolations.integral(
                    Ihat,
                    t[i-1],
                    t[i]
                )

                G_current += quadgk(
                    s -> Ihat(s)^2,
                    t[i-1],
                    t[i]
                )[1]

                push!(F_values, F_current)
                push!(G_values, G_current)
            end
        end

        Fhat = build_I_interpolant(t, F_values, method; order=order)
        Ghat = build_I_interpolant(t, G_values, method; order=order)

        for k in 1:K
            # General weak LHS, including boundary term
            phi, dphi = get_testing_function(t, k, K, testing_function; m)

            # General weak LHS, including boundary term
            if testing_function == :chebyshev_U
                Y[k] = chebyshev_U_Y(Ihat, t0, tT, k)
            else
                boundary = phi(tT) * I_data[end] - phi(t0) * I_data[1]
                Y[k] = boundary - quadgk(x -> dphi(x) * Ihat(x), t0, tT)[1]
            end

            W1[k] = quadgk(x -> phi(x), t0, tT)[1]
            W2[k] = quadgk(x -> phi(x) * Ihat(x), t0, tT)[1]
            W3[k] = quadgk(x -> phi(x) * (Ihat(x)^2 - I0^2), t0, tT)[1]
            W4[k] = quadgk(x -> phi(x) * Fhat(x), t0, tT)[1]
            W5[k] = quadgk(x -> phi(x) * Ghat(x), t0, tT)[1]
            W6[k] = quadgk(x -> phi(x) * Fhat(x)^2, t0, tT)[1]
        end

    else

        F = Integrate.integrate(t, I_data, method)
        G = Integrate.integrate(t, I_data.^2, method)

        for k in 1:K
            phi, dphi = Measure.measure_sine(t, k)

            Y[k]  = - Integrate.integrate(t, dphi .* I_data, method)[end]

            W1[k] = Integrate.integrate(t, phi, method)[end]
            W2[k] = Integrate.integrate(t, phi .* I_data, method)[end]
            W3[k] = Integrate.integrate(t, phi .* (I_data.^2 .- I0^2), method)[end]
            W4[k] = Integrate.integrate(t, phi .* F, method)[end]
            W5[k] = Integrate.integrate(t, phi .* G, method)[end]
            W6[k] = Integrate.integrate(t, phi .* (F.^2), method)[end]
        end
    end

    return Y, W1, W2, W3, W4, W5, W6
end


function get_blocks(
    I_data::Vector{Float64},
    t::Vector{Float64},
    method::String
)
    t0 = t[1]

    if !(in_exception(method))
        I0 = I_data[1]

        B1 = 1
        B2 = Integrate.integrate(t, I_data, method)
        B3 = Integrate.integrate(t, I_data.^2 .- I0^2, method)
        B4 = t .* B2 .- Integrate.integrate(t, t .* I_data, method)
        B5 = t .* Integrate.integrate(t, I_data.^2, method) .- Integrate.integrate(t, t .* (I_data.^2), method)
        B6 = Integrate.integrate(t, (B2).^2, method)

        return I0, B1, B2, B3, B4, B5, B6

    elseif method in ["QSpline_GK", "CSpline_GK", "Akima_GK"]
        Ihat = build_I_interpolant(t, I_data, method)
        I0 = Ihat(t0)

        B1 = 1
        B2 = [DataInterpolations.integral(Ihat, t0, x) for x in t]
        B3 = cumulative_quadgk(s -> (Ihat(s)^2 - I0^2), t)
        B4 = t .* B2 .- cumulative_quadgk(s -> (s * Ihat(s)), t)
        B5 = t .* cumulative_quadgk(s -> Ihat(s)^2, t) .- cumulative_quadgk(s -> (s * Ihat(s)^2), t)

        # TODO: Check these two approaches
        # F(x) = DataInterpolations.integral(Ihat, t0, x)
        # B6 = [quadgk(s -> (F(s)^2), t0, x)[1] for x in t]

        B2hat = build_I_interpolant(t, B2, method)
        B6 = cumulative_quadgk(s -> B2hat(s)^2, t)

        return I0, B1, B2, B3, B4, B5, B6

    else
        error("Unknown integration method $method")
    end
end


function L_hat(paras, I0, W1, W2, W3, W4, W5, W6)
    """
    Vector of length K, containing the RHS of the equation computed by the formula
    When computing residual, instead, we do weak_I_hat - Y
    """
    α_eff  = paras[1]
    σ_eff  = paras[2]
    γ_eff  = paras[3]
    S0_eff = paras[4]
    E0_eff = paras[5]

    C1 = σ_eff * (E0_eff + I0) .* W1
    C2 = - (γ_eff + σ_eff) .* W2
    C3 = - 0.5 * α_eff .* W3
    C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* W4
    C5 = - α_eff * (γ_eff + σ_eff) .* W5
    C6 = - 0.5 * α_eff * σ_eff * γ_eff .* W6

    # Here we dont have I0 anymore? (since we do differentiate and then integrate)
    return C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
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


    Y, W1, W2, W3, W4, W5, W6 = get_weak_blocks(
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
            W3[k],
            W4[k],
            W5[k],
            W6[k]
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
    println("W4[end] = ", W4[end])
    println("W5[end] = ", W5[end])
    println("W6[end] = ", W6[end])

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
    Y, W1, W2, W3, W4, W5, W6 = get_weak_blocks(I_data, t, K, method, testing_function; m=m, order=order)

    s = [
        norm(Y),
        norm(W1),
        norm(W2),
        norm(W3),
        norm(W4),
        norm(W5),
        norm(W6)
    ]

    powers = [0, 1, 1, 1, 2, 2, 3]

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


struct RootRecord
    index::Int

    # HC
    hc_root::Vector{Float64}
    hc_rss::Float64

    # clipping information
    clipped::Bool
    clipped_root::Vector{Float64}

    # LS stage
    ls_success::Bool
    ls_root::Union{Vector{Float64}, Nothing}
    ls_rss::Union{Float64, Nothing}
    ls_iterations::Union{Int, Nothing}

    # if LS failed
    error_message::Union{String, Nothing}
end


function RootRecord(;
    index,
    hc_root,
    hc_rss,
    clipped,
    clipped_root,
    ls_success,
    ls_root=nothing,
    ls_rss=nothing,
    ls_iterations=nothing,
    error_message=nothing
)
    return RootRecord(
        index,
        hc_root,
        hc_rss,
        clipped,
        clipped_root,
        ls_success,
        ls_root,
        ls_rss,
        ls_iterations,
        error_message
    )
end


function print_root_info(r::RootRecord)

    println("======================================")
    println("Root #$(r.index)")
    println("======================================")

    println("HC result:")
    println("  params = ", r.hc_root)
    println("  RSS    = ", r.hc_rss)

    println()

    println("Clipping:")
    println("  happened = ", r.clipped)

    if r.clipped
        println("  before = ", r.hc_root)
        println("  after  = ", r.clipped_root)
    end

    println()

    println("LS result:")
    println("  success = ", r.ls_success)

    if r.ls_success
        println("  params = ", r.ls_root)
        println("  RSS    = ", r.ls_rss)
        println("  iterations = ", r.ls_iterations)

        println()

        println("Improvement:")
        println("  ΔRSS = ", r.hc_rss - r.ls_rss)

    else
        println("  FAILED")
        println("  error = ", r.error_message)
    end

    println()
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

    times["select K T"] = time() - time_start

    time_start = time()
    t_scaled = t ./ T
    Y, W1, W2, W3, W4, W5, W6 = get_weak_blocks(I_data, t_scaled, K, method, testing_function; m=m, order=order, plot_Ihat=plot_Ihat)
    times["get weak blocks"] = time() - time_start

    time_start = time()
    B = get_blocks(I_data, t_scaled, method)
    times["get blocks"] = time() - time_start

    time_start = time()
    I0 = I_data[1]

    iteration_counts = 0
    function model(x, p)
        iteration_counts += 1
        return Logic.I_hat(p, B..., x)
    end

    Lhat = L_hat(vars, I0, W1, W2, W3, W4, W5, W6)
    J = sum((Lhat .- Y) .^ 2)

    system_eqs = differentiate(J, vars)
    C = System(system_eqs, variables=vars)

    result = HomotopyContinuation.solve(C, show_progress=false)
    real_results = real_solutions(result)

    if isempty(real_results)
        error("No real HC solution found for SEIR weak form.")
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
        hc_rss = Logic.get_RSS(I_data, Logic.I_hat(r, B..., t_scaled))
        push!(hc_rss_list, hc_rss)

        # Make sure the starting point is inside the LS bounds
        clipped_root = min.(max.(hc_root, lb_scaled), ub_scaled)

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
            ls_rss = Logic.get_RSS(I_data, Logic.I_hat(ls_root, B..., t_scaled))

            record = RootRecord(
                index = i,
                hc_root = hc_root,
                hc_rss = hc_rss,

                clipped = clipped,
                clipped_root = clipped_root,

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
        printstyled("===== HC_LS_weak SEIR Results =====\n", color = :magenta, bold = true)
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

            RSS_Ihat_Idata_true = Logic.get_RSS(I_data, Logic.I_hat(true_scaled, B..., t_scaled))
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

#        println("\nALL real results -- #", length(real_results))
#        for r in real_results
#            printstyled("------------\n", color = :blue)
#            r_physical = Logic.to_physical(r, T)
#            println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(r, B..., t_scaled)))
#
#            if true_vals !== nothing
#                println("parameter error ", Logic.get_param_error(r_physical, true_vals))
#            end
#
#            printstyled("------------\n", color = :blue)
#        end
#
#        println("\nALL final results -- #", length(final_results_scaled))
#        for (ind, r) in enumerate(final_results_scaled)
#            printstyled("------------\n", color = :blue)
#
#            if r == best_result_scaled
#                printstyled("best result!\n", color=:yellow)
#            end
#            r_physical = Logic.to_physical(r, T)
#
#            iter_r = iteration_count_list[ind]
#            println("model evaluations = ", iter_r)
#            println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(r, B..., t_scaled)))
#
#            if true_vals !== nothing
#                println("parameter error ", Logic.get_param_error(r_physical, true_vals))
#            end
#            printstyled("------------\n", color = :blue)
#        end


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
                println("RSS_Ihat_Idata ", Logic.get_RSS(I_data, Logic.I_hat(fit.param, B..., t_scaled)))
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
