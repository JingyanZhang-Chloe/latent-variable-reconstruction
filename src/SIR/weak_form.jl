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


function get_weak_blocks(I_data::Vector{Float64}, t::Vector{Float64}, method::String, K::Int)
    # Since we are differentiating, maybe it is better to write in the original form
    Y  = zeros(K)   # weak left-hand side ∫ phi I'
    W1 = zeros(K)   # ∫ phi I
    W2 = zeros(K)   # ∫ phi I^2
    W3 = zeros(K)   # ∫ phi I F

    # Using integration by parts:
    # ∫ phi I' = [phi I]_0^T - ∫ phi' I
    # If we ensure [phi I]_0^T = 0, then LHS = - ∫ phi' I

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
    true_vals=Value.true_vals
)
    """
    No time rescaling
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """

    Y, W1, W2, W3 = get_weak_blocks(I_data, t, method, K)
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

    final_results = Vector{Float64}[]

    xdata = collect(1:K)

    for r in real_results
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

        catch e
            @warn "curve_fit failed for initial point" p0 exception=e
        end
    end

    if isempty(final_results)
        error("No valid LS-refined solutions found.")
    end

    best_result, RSS = best_solution_weak(final_results, Y, I0, W1, W2, W3)
    parameter_err = Logic.get_param_error(best_result, true_vals)

    println("=== HC_LS_weak Results ===")
    println("Method used: ", method)
    println("Number of test functions K: ", K)

    println("\nBest parameter estimates:")
    for (var, val) in zip(vars, best_result)
        println(var, " = ", val)
    end

    println("\nResidual sum of squares (RSS_Lhat_L(Y)): ", RSS)
    println("\nParameter error: ", parameter_err)

    return best_result, RSS, parameter_err
end
