# Measure.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 22/06/2026
=#


module Measure

    function measure_sine(t::Vector, k::Int)
        """
        Return a vector storing values of appropriate function phi at each time in the time vector t
        Normalize to make sure φ[0] = φ[T] = 0
        φ_k = sin(kπ (t-t0)/L)
        """

        t0 = t[1]
        T = t[end]
        L = T - t0
        x = (t .- t0) ./ L

        phi = sin.(k * π .* x)
        dphi = (k * π / L) .* cos.(k * π .* x)

        return phi, dphi
    end


    function measure_sine_function(t::AbstractVector{<:Real}, k::Int)
        t0 = t[1]
        T = t[end]
        L = T - t0

        phi(s) = sin(k * π * (s - t0) / L)
        dphi(s) = (k * π / L) * cos(k * π * (s - t0) / L)

        return phi, dphi
    end

end
