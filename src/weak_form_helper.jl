# weak_form_helper.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 21/07/2026
=#

using DataInterpolations
using RegularizationTools
using Profile
using ProfileView


function in_exception(me::String)
    return !(me in ["T", "S", "S_uniform"])
end


function build_I_interpolant(t::Vector{Float64}, I_data::Vector{Float64}, method::String;
    order::Union{Int,Nothing}=nothing,
    n_control_points::Union{Int, Nothing}=nothing,
    d_smooth::Union{Int, Nothing}=nothing,
    λ::Union{Int, Nothing}=nothing
)
    if method in ["QSpline_GK", "Qspline_exact"]
        return QuadraticSpline(I_data, t)

    elseif method in ["CSpline_GK", "CSpline_exact"]
        return CubicSpline(I_data, t)

    elseif method in ["Akima_GK", "Akima_exact"]
        return AkimaInterpolation(I_data, t)

    elseif method in ["BSpline_GK"]
        if order === nothing
            error("B_Spline requires interpolation order")
        end

        B = BSplineInterpolation(I_data, t, order, :ArcLen, :Average)
        return B

    elseif method in ["BSplineApprox"]
        if order === nothing
            error("B_SplineApprox requires interpolation order")
        end

        if n_control_points === nothing
            error("B_SplineApprox requires n_control_points")
        end

        return BSplineApprox(
            I_data,
            t,
            order,
            n_control_points,
            :Uniform,
            :Average
        )

    elseif method in ["RegularizationSmooth"]
        if d_smooth === nothing
            error("RegularizationSmooth requires derivative order d_smooth")
        end

        if λ === nothing
            error("RegularizationSmooth requires the regularization (smoothing) parameter λ")
        end

        return RegularizationSmooth(
            I_data,
            t,
            d_smooth;
            λ = λ,
            alg = :fixed
        )

    elseif method in ["PCHIP"]
        return PCHIPInterpolation(I_data, t)

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


"""
Compute [quadgk(s -> Ihat(s)^power, t0, x)[1] for x in t]
"""
function cumulative_quadgk(
    integrand,
    t::Vector{Float64}
)
    F_values = Float64[]
    F_current = 0.0

    push!(F_values, F_current)

    for i in 2:length(t)
        F_current += quadgk(
            integrand,
            t[i-1],
            t[i]
        )[1]

        push!(F_values, F_current)
    end

    return F_values
end


struct RootRecord
    index::Int

    # HC
    hc_root::Vector{Float64}
    hc_rss::Float64

    # clipping information
    clipped::Bool
    clipped_root::Vector{Float64}
    clipped_rss::Union{Float64, Nothing}

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
    clipped_rss,
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
        clipped_rss,
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
        println(" Root after clipping = ", r.clipped_root)
        println(" RSS after clipping = ", r.clipped_rss)
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