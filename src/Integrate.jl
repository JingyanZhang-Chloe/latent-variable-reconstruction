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


    function integrate(t::Vector, y::Vector, method::String)
        if method == "T"
            return cumul_integrate(t, y)
        elseif method == "S"
            return cumintegrate(t, y)
        elseif method == "S_uniform"
            return cumintegrate_simpson_uniform(t, y)
        else
            @error "method must be T, S, or S_uniform"
        end
    end

end