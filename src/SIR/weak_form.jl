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
using Printf
using LinearAlgebra
using QuadGK
using DataInterpolations


function in_exception(me::String)
    return (me in ["S_improved", "S_formula_improved", "QSpline_GK", "CSpline_GK", "Akima_GK", "Qspline_exact", "CSpine_exact", "Akima_exact"])
end


function build_I_interpolant(t::Vector{Float64}, I_data::Vector{Float64}, method::String)
    if method in ["QSpline_GK", "Qspline_exact"]
        return QuadraticSpline(I_data, t)

    elseif method in ["CSpline_GK", "CSpine_exact"]
        return CubicSpline(I_data, t)

    elseif method in ["Akima_GK", "Akima_exact"]
        return AkimaInterpolation(I_data, t)

    else
        error("Unknown spline-GK method: $method")
    end
end


function local_poly_coeffs(Ihat, a, b, degree::Int)
    """
    QSpline has degree 2
    CSpline and Akima have degree 3 (since C2)

    We are storing the coefficients of q(z)
    Then Ihat in [a, b] => Ihat(x) = q(x-m)
    """

    m = (a + b) / 2
    h = (b - a) / 2

    c0 = Ihat(m)
    c1 = DataInterpolations.derivative(Ihat, m, 1)
    c2 = 0.5 * DataInterpolations.derivative(Ihat, m, 2)

    if degree == 2
        return [c0, c1, c2, 0], m
    elseif degree == 3
        c3 = (Ihat(b) - Ihat(a) - 2 * c1 * h) / 2 * h^3

        return [c0, c1, c2, c3], m
    else
        error("Only degree 2 or 3 supported for now")
    end
end


function poly_mul(p_coeffs::Vector{Float64}, q_coeffs::Vector{Float64})
    """
    Multiply two polynomials, given their coefficients as Vector
    Return the resulting polynomial coefficients Vector
    """
    r = zeros(Float64, length(p_coeffs) + length(q_coeffs) - 1)

    for i in eachindex(p_coeffs)
        for j in eachindex(q_coeffs)
            r[i + j - 1] += p_coeffs[i] * q_coeffs[j]
        end
    end

    return r
end


function get_F_coeffs(qcoeffs::Vector{Float64}, h, Fa)
    """
    Giving the coeff of Ihat, compute on the interval [a, b]
    F = ∫ Ihat(s) ds
    Return the coefficients Vector of F, since it is also a polynomial
    Assuming we know F(a)

    F(s) = ∫_a^s Ihat(x)dx
    => (z = x - m) recall m = a+b/2 and h = b-a/2
    F(s) = ∫_(a-m)^(s-m) [c0 + c1 z + c2 z^2 + c3 z^3] dz
    =>
    F(s) = ∫_(-h)^(s-m) [c0 + c1 z + c2 z^2 + c3 z^3] dz
    """

    deg = length(qcoeffs) - 1
    Fcoeffs = zeros(Float64, deg + 2)

    Fcoeffs[1] = Fa
    for n in 0:deg
        # For each term of q --- c_n * z^n --- integrate: c_n 1/n+1 * ((s-m)^(n+1) - (-h)^(n+1))
        # If we do u = s-m
        # So F(s) = F(u + m) = ~F(u) = poly we are computing rn
        # So we have a constant term: - c_n * 1/n+1 * (-h)^(n+1)
        # And a z^(n+1) term with coeff: c_n * 1/n+1
        c = qcoeffs[n + 1]

        # constant part from - c * (-h)^(n+1)/(n+1)
        Fcoeffs[1] -= c * (-h)^(n + 1) / (n + 1)

        # coefficient of z^(n+1)
        Fcoeffs[n + 2] += c / (n + 1)
    end

    return Fcoeffs
end


function cumulative_F(Fa, a, b)
    """
    Compute F(b)
    """
end


function trig_moments(n, basis::Symbol, a, b)
    """
    Compute ∫ z^n sin or ∫ z^n cos
    """
end


function get_testing_function(t::Vector{Float64}, k::Int, K::Int, testing_function::Symbol, if_function::Bool=true; m::Union{Int, Nothing})
    if testing_function == :sin
        if if_function
            phi, dphi = Measure.measure_sine_function(t, k)
        else
            phi, dphi = Measure.measure_sine(t, k)
        end

    elseif testing_function == :bump
        if if_function
            phi, dphi, _ = Measure.measure_bump_function(t, k, K)
        else
            phi, dphi, _ = Measure.measure_bump(t, k, K)
        end

    elseif testing_function == :hartley
        if if_function
            phi, dphi, _ = Measure.measure_hartley_function(t, k, K)
        else
            phi, dphi, _ = Measure.measure_hartley(t, k, K)
        end

    elseif testing_function == :polynomial
        if m === nothing
            error("Polynomial test functions require degree m")
        end

        if if_function
            phi, dphi, _ = Measure.measure_polynomial_function(t, k, K, m)
        else
            phi, dphi, _ = Measure.measure_polynomial(t, k, K, m)
        end

    elseif testing_function == :chebyshev_T
        error("boundary issue")

    elseif testing_function == :chebyshev_U
        if if_function
            phi, dphi = Measure.chebyshev_U_function(t, k)
        else
            phi, dphi = Measure.chebyshev_U(t, k)
        end

    else
        error("not implemented yet heihei")
    end
end


using QuadGK

function chebyshev_U_Y(
    Ihat,
    t0,
    tT,
    k
)
    """
    This is for IHAT is a FUNCTION!

    Y = boundary - quadgk(x -> dphi(x) * Ihat(x), t)[1]

    If chebyshev U, then we know phi(-1) = phi(1) = 0, boundary = 0
    Y = - quadgk(x -> dphi(x) * Ihat(x), t)[1]

    change of variable x = cosθ
    Then ∫ from π to 0
    θ = π corresponds to t0
    θ = 0 corresponds to tT

    x(t) = (2t-(t0+tT))/(tT-t0)
    t(θ) = [cosθ (tT - t0) + (t0 + tT)] / 2
    x(t(θ)) = cosθ

    Since phi(t) = sin((k+1)*acos(x(t)))
    phi(t(θ)) = sin((k+1)θ)

    dphi(t(θ)) = dphi/dt (t(θ))
    We have dphi/dθ = (k+1)cos((k+1)θ)
    by chain rule dphi(t(θ)) = dphi/dt (t(θ)) = dphi/dθ * dθ/dt = (k+1)cos((k+1)θ) * - 2/Lsinθ
    dt = - Lsinθ/2 dθ

    Hence the integral Y = - quadgk(x -> dphi(x) * Ihat(x), t)[1]
    ==> Y = - quadgk(θ -> Ihat(x) * {[(k+1)cos((k+1)θ) * - 2/Lsinθ] * - Lsinθ/2}, π, 0)
    ==> Y = quadgk(θ -> Ihat(x) * {(k+1)cos((k+1)θ)}, 0, π)
    """
    L = tT - t0

    t_from_theta(θ) =
        (t0 + tT) / 2 + (L / 2) * cos(θ)

    m = k + 1

    return m * quadgk(
        θ -> Ihat(t_from_theta(θ)) * cos(m * θ),
        0,
        π
    )[1]
end


function chebyshev_U_Y_vector(
    I_data::Vector{Float64},
    t::Vector{Float64},
    k,
    method
)
    """
    This is for IHAT is a FUNCTION!
    """
    t0 = t[1]
    tT = t[end]
    m = k + 1

    x = clamp.(
        (2 .* t .- (t0 + tT)) / (tT - t0),
        -1.0,
        1.0,
    )

    # t increases, but acos(x) decreases from π to 0
    theta = reverse(acos.(x))
    I_theta = reverse(I_data)

    measure_theta(θ) = cos(m * θ)

    if method == "S_improved"
        return m * Integrate.integrate(
            theta,
            I_theta,
            method;
            measure = measure_theta,
        )
    else
        measure_theta_vector = measure_theta.(theta)

        return m * Integrate.integrate(
            theta,
            measure_theta_vector .* I_theta,
            method
        )[end]
    end
end


function get_weak_blocks(I_data::Vector{Float64}, t::Vector{Float64}, K::Int, method, testing_function::Symbol; m::Union{Int, Nothing}=nothing)
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
        Ihat = build_I_interpolant(t, I_data, method)
        # So Ihat is Ihat(x) = c0 + c1 * z + c2 * z^2 + c3 * z^3

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
    testing_function::Symbol
)
    println("======================================")
    println("Weak block analysis")
    println("K = ", K)
    println("Testing function: $testing_function")
    println("======================================")
    for method in method_list
        println()
        println("-------------------------------------")
        printstyled("method = ", method, color=:yellow, bold=true)
        println()

        Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method, testing_function)

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
        # Table 1: row norms and cond value
        # ------------------------------------------------------------
        println()
        @printf("%4s %14s %14s %14s %14s %14s\n",
                "k", "|Y|", "|W1|", "|W2|", "|W3|", "row norm")

        row_norms = zeros(K)

        for k in 1:K
            row_norms[k] = sqrt(Y[k]^2 + W1[k]^2 + W2[k]^2 + W3[k]^2)

            @printf("%4d %14.4e %14.4e %14.4e %14.4e %14.4e\n",
                    k,
                    abs(Y[k]),
                    abs(W1[k]),
                    abs(W2[k]),
                    abs(W3[k]),
                    row_norms[k])
        end

        println()
        println("Summary:")
        println("min row norm = ", minimum(row_norms))
        println("max row norm = ", maximum(row_norms))
        println("max/min row norm = ", maximum(row_norms) / minimum(row_norms))

        A = hcat(W1, W2, W3)

        if K >= 3
            println("condition number of [W1 W2 W3] = ", cond(A))
        else
            println("condition number skipped because K < 3")
        end
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


function select_T_weak(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method::String,
    testing_function::Symbol;
    m,
    m_min::Int = -6,
    m_max::Int = 6,
)
    Y, W1, W2, W3 = get_weak_blocks(I_data, t, K, method, testing_function; m=m)

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


function HC_LS_weak(
    t::Vector{Float64},
    I_data::Vector{Float64},
    vars::Vector,
    method::String,
    testing_function::Symbol;
    K::Int = 8,
    true_vals=Value.true_vals,
    if_print=true,
    m=10
)
    """
    YES time rescaling
    No complicated projection to bounds after HC
    Still make initial points in bounds before LS
    """
    T, _ = select_T_weak(I_data, t, K, method, testing_function; m=m)
    t_scaled = t ./ T
    Y, W1, W2, W3 = get_weak_blocks(I_data, t_scaled, K, method, testing_function; m=m)

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

    if if_print
        if in_exception(method)
            B = Logic.get_blocks(I_data, t_scaled, "S")
        else
            B = Logic.get_blocks(I_data, t_scaled, method)
        end
        Ihat_best = Logic.I_hat(best_result_scaled, B...)

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
        println("Residual sum of squares (RSS_Ihat_Idata): ", Logic.get_RSS(Ihat_best, I_data))

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

    return best_result, RSS, parameter_err
end


"""
NOT UPDATED
This version uses NO time rescaling
"""
function _HC_LS_weak(
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
