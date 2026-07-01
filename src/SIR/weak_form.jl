# weak_form.jl
# Julia Script

#=
Description: Weak form test for SIR integral elimination
Author: zhangjingyan
Date: 22/06/2026
=#

include("SIRModels.jl")
include("../Measure.jl")

using .Value
using .Logic
using .Integrate
using .Measure
using HomotopyContinuation
using LsqFit
using Random


function get_weak_blocks(I_data::Vector{Float64}, t::Vector{Float64}, K::Int, method)
    # Since we are differentiating, maybe it is better to write in the original form
    Y  = zeros(K)   # weak left-hand side ∫ phi I'
    W1 = zeros(K)   # ∫ phi I
    W2 = zeros(K)   # ∫ phi I^2
    W3 = zeros(K)   # ∫ phi I F

    # Using integration by parts:
    # ∫ phi I' = [phi I]_0^T - ∫ phi' I
    # If we ensure [phi I]_0^T = 0, then LHS = - ∫ phi' I

    if method == "S_improved"
        F = Integrate.integrate(t, I_data, "S")

        for k in 1:K
            phi, dphi = Measure.measure_sine_function(t, k)

            # General weak LHS, including boundary term
            boundary = phi(t[end]) * I_data[end] - phi(t[1]) * I_data[1]
            Y[k] = boundary - Integrate.integrate(t, I_data, method; measure=dphi)

            W1[k] = Integrate.integrate(t, I_data, method; measure=phi)
            W2[k] = Integrate.integrate(t, (I_data .^ 2), method; measure=phi)
            W3[k] = Integrate.integrate(t, I_data .* F, method; measure=phi)
        end
    else
        F = Integrate.integrate(t, I_data, method)

        for k in 1:K
            phi, dphi = Measure.measure_sine(t, k)

            # General weak LHS, including boundary term
            boundary = phi[end] * I_data[end] - phi[1] * I_data[1]
            Y[k] = boundary - Integrate.integrate(t, dphi .* I_data, method)[end]

            W1[k] = Integrate.integrate(t, phi .* I_data, method)[end]
            W2[k] = Integrate.integrate(t, phi .* (I_data .^ 2), method)[end]
            W3[k] = Integrate.integrate(t, phi .* I_data .* F, method)[end]
        end
    end

    return Y, W1, W2, W3
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


function HC_LS_weak(
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
        if method == "S_improved"
            B = Logic.get_blocks(I_data, t, "S")
        else
            B = Logic.get_blocks(I_data, t, method)
        end
        Ihat_best = Logic.I_hat(best_result, B...)

        printstyled("=== HC_LS_weak SIR Results ===\n", color = :magenta, bold = true)
        println("Method used: ", method)
        println("Number of test functions K: ", K)

        println("\nBest parameter estimates:")
        for (var, val) in zip(vars, best_result)
            println(var, " = ", val)
        end

        println("\nResidual sum of squares (RSS_Lhat_L(Y)): ", RSS)
        println("Residual sum of squares (RSS_Ihat_Idata): ", Logic.get_RSS(Ihat_best, I_data))

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
