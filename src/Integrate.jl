# Integrate.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 22/06/2026
=#

module Integrate

    function cumintegrate(x::AbstractVector, y::AbstractVector)
        n = length(x)
        T = promote_type(eltype(x), eltype(y))
        output = zeros(T, n)

        if n == 1
            error("cumintegrate requires at least 2 points")
        end

        if n == 2
            output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2
            return output
        end

        for i in 3:2:n
            x1 = x[i-2]
            x2 = x[i-1]
            x3 = x[i]
            y1 = y[i-2]
            y2 = y[i-1]
            y3 = y[i]

            h1 = x2 - x1
            h2 = x3 - x2
            h_total = x3 - x1

            # use the standard Simpson's 1/3 rule
            # from http://www.msme.us/2017-2-1.pdf formula 6
            output[i] = output[i-2] + (h_total / 6) * (
                (2 - h2 / h1) * y1 +
                (h_total^2 / (h1 * h2)) * y2 +
                (2 - h1 / h2) * y3
            )

            # to compute output[i-1] we use the formula for scipy.integrate.cumulative_simpson
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html#rb3a817c91225-2
            # from http://www.msme.us/2017-2-1.pdf formula 8
            output[i-1] = output[i-2] + (h1 / 6) * (
                (3 - h1 / h_total) * y1 +
                (3 + h1^2 / (h2 * h_total) + h1 / h_total) * y2 -
                (h1^2 / (h2 * h_total)) * y3
            )
        end

        if iseven(n) && n >= 3
            # Use the last 3 points
            x1, x2, x3 = x[n-2], x[n-1], x[n]
            y1, y2, y3 = y[n-2], y[n-1], y[n]
            h1, h2 = x2 - x1, x3 - x2
            h_total = x3 - x1

            # use formula 8 to compute the last point integration
            # notice we need to do for these three points: total_simpson - first_half
            total_simpson = (h_total / 6) * (
                (2 - h2 / h1) * y1 +
                (h_total^2 / (h1 * h2)) * y2 +
                (2 - h1 / h2) * y3
            )

            first_half = (h1 / 6) * (
                (3 - h1 / h_total) * y1 +
                (3 + h1^2 / (h2 * h_total) + h1 / h_total) * y2 -
                (h1^2 / (h2 * h_total)) * y3
            )

            output[n] = output[n-1] + (total_simpson - first_half)
        end

        return output
    end


    function cumintegrate_simpson_uniform(x::AbstractVector, y::AbstractVector)
        n = length(x)
        T = promote_type(eltype(x), eltype(y))
        output = zeros(T, n)

        if n == 1
            error("cumintegrate requires at least 2 points")
        end

        if n == 2
            output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2
            return output
        end

        output[1] = zero(T)
        output[2] = (x[2] - x[1]) * (y[1] + y[2]) / 2

        for i in 3:2:n
            x1 = x[i-2]
            x2 = x[i-1]
            x3 = x[i]
            y1 = y[i-2]
            y2 = y[i-1]
            y3 = y[i]

            h1 = x2 - x1
            h2 = x3 - x2
            h_total = x3 - x1

            # use the standard Simpson's 1/3 rule
            # from http://www.msme.us/2017-2-1.pdf formula 6
            output[i] = output[i-2] + (h_total / 6) * (
                (2 - h2 / h1) * y1 +
                (h_total^2 / (h1 * h2)) * y2 +
                (2 - h1 / h2) * y3
            )

            # to compute output[i-1] we use the formula for scipy.integrate.cumulative_simpson
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html#rb3a817c91225-2
            # from http://www.msme.us/2017-2-1.pdf formula 8
            output[i-1] = output[i-2] + (h1 / 6) * (
                (3 - h1 / h_total) * y1 +
                (3 + h1^2 / (h2 * h_total) + h1 / h_total) * y2 -
                (h1^2 / (h2 * h_total)) * y3
            )
        end

        if iseven(n)
            # Use the last 2 points
            h = x[n] - x[n-1]
            trap_step = h * (y[n-1] + y[n]) / 2
            output[n] = output[n-1] + (trap_step)
        end

        return output
    end


    using QuadGK
    """
    quadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing, eval_segbuf=nothing)

    The algorithm is an adaptive Gauss-Kronrod integration technique:
    the integral in each interval is estimated using a Kronrod rule (2*order+1 points)
    and the error is estimated using an embedded Gauss rule (order points).
    The interval with the largest error is then subdivided into two intervals and the process is repeated until the desired error tolerance is achieved.
    """

    function quadratic_lagrange_integral(
        measure::Function,
        a, b,
        x0,
        x1,
        x2,
        y0,
        y1,
        y2,
        rtol
    )
        """
        given x, y, measure
        return ∫ from x0 to x2 of measure * y wrt x
        """

        if x0 == x1 || x0 == x2 || x1 == x2
            error("Interpolation nodes must be distinct")
        end

        # Here based on the Lagrange interpolation formula for approximating y (first part of Simpson method)
        # L = y0*l0 + y1*l1 + y2*l2 the approximation of f
        l0(x) = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
        l1(x) = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
        l2(x) = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))

        # So target integration is
        # ∫ [y0*l0 + y1*l1 + y2*l2] * measure
        # y0 * ∫ l0*measure + y1 * ∫ l1*measure + y2 * ∫ l2*measure
        # Since now l * measure is no longer the quadratic function approximated in original Simpson method, we cannot directly apply the formula?

        A0, _ = quadgk(x -> l0(x) * measure(x), a, b; rtol=rtol)
        A1, _ = quadgk(x -> l1(x) * measure(x), a, b; rtol=rtol)
        A2, _ = quadgk(x -> l2(x) * measure(x), a, b; rtol=rtol)

        return A0 * y0 + A1 * y1 + A2 * y2
    end


    function cumintegrate_improved(x::AbstractVector, y::AbstractVector, measure::Function, rtol)::Float64
        """
        compute the integration of measure * y wrt x
        since for now we only need the integrating value on the full interval x, only compute that value
        """
        n = length(x)

        if n == 1
            error("cumintegrate requires at least 2 points")
        end

        if n == 2
            x0 = x[1]
            x1 = x[2]
            y0 = y[1]
            y1 = y[2]

            l0(x) = (x - x1) / (x0 - x1)
            l1(x) = (x - x0) / (x1 - x0)

            A0, _ = quadgk(x -> l0(x) * measure(x), x0, x1; rtol=rtol)
            A1, _ = quadgk(x -> l1(x) * measure(x), x0, x1; rtol=rtol)

            return A0 * y0 + A1 * y1
        end

        total = 0.0

        # Use quadratic blocks
        # (x1,x2,x3), then (x3,x4,x5), then (x5,x6,x7), ...
        last_full_point = isodd(n) ? n : n - 1

        for i in 1:2:(last_full_point - 2)
            total += quadratic_lagrange_integral(
                measure,
                x[i], x[i+2],
                x[i], x[i+1], x[i+2],
                y[i], y[i+1], y[i+2],
                rtol
            )
        end

        # If n is even, one last interval remains
        # we approximate y using the last three points, but integrate only from x[n-1] to x[n]
        if iseven(n)
            total += quadratic_lagrange_integral(
                measure,
                x[n-1], x[n],
                x[n-2], x[n-1], x[n],
                y[n-2], y[n-1], y[n],
                rtol
            )
        end

        return total
    end


    function integrate(t::Vector, y::Vector, method::String; measure::Union{Function, Nothing}=nothing, rtol=1e-10)
        if method == "T"
            return cumul_integrate(t, y)
        elseif method == "S"
            return cumintegrate(t, y)
        elseif method == "S_uniform"
            return cumintegrate_simpson_uniform(t, y)
        elseif method == "S_improved"
            if measure === nothing
                error("S_improved require a measure function")
                return
            else
                return cumintegrate_improved(t, y, measure, rtol)
            end
        else
            error("method must be T, S, S_uniform, or S_improved")
        end
    end

end