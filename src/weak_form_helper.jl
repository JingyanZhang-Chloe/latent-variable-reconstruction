# weak_form_helper.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 21/07/2026
=#

function in_exception(me::String)
    return (me in ["S_improved", "S_formula_improved", "QSpline_GK", "CSpline_GK", "Akima_GK", "Qspline_exact", "CSpine_exact", "Akima_exact", "BSpline_GK"])
end


function build_I_interpolant(t::Vector{Float64}, I_data::Vector{Float64}, method::String; order::Union{Int,Nothing}=nothing)
    if method in ["QSpline_GK", "Qspline_exact"]
        return QuadraticSpline(I_data, t)

    elseif method in ["CSpline_GK", "CSpine_exact"]
        return CubicSpline(I_data, t)

    elseif method in ["Akima_GK", "Akima_exact"]
        return AkimaInterpolation(I_data, t)

    elseif method in ["BSpline_GK"]
        if order === nothing
            error("B_Spline requires interpolation order")
        end

        B = BSplineInterpolation(I_data, t, order, :ArcLen, :Average)
        return B
    else
        error("Unknown spline-GK method: $method")
    end
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

    m = k

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
    m = k

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
