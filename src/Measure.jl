# Measure.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 22/06/2026
=#


module Measure

    # ====================================================================================================
    # Some helper functions
    # ====================================================================================================

    # For now we used Trapz to compute integration (when computing L2 norm), we give the function as Vector!!
    # This is what the paper used?
    # Later we could try Simpson
    function trapz(t::AbstractVector, y::AbstractVector)
        @assert length(t) == length(y)
        s = 0.0
        for i in 1:length(t)-1
            s += 0.5 * (t[i+1] - t[i]) * (y[i] + y[i+1])
        end
        return s
    end

    function l2_scale(t::AbstractVector, phi_raw::AbstractVector)
        """
        Give the unnormalized testing function φ, compute C so that Cφ has L2 norm 1
        """
        nrm2 = trapz(t, phi_raw .^ 2)
        if nrm2 <= eps()
            println("phi min = ", minimum(phi_raw))
            println("phi max = ", maximum(phi_raw))
            println("phi norm = ", trapz(t, phi_raw.^2))
            error("Cannot normalize: test function is almost zero on this grid")
        end
        return 1.0 / sqrt(nrm2)
    end

    # Given k as the k-th WANDy function we wanna compute
    # Given K (total weak forms) and t, and support range a
    # Choose center c_k and make sure [c_k-a, c_k+a] stays inside [t0,T]
    function support_center(t::AbstractVector{<:Real}, k::Int, K::Int, a::Real)
        @assert 1 <= k <= K

        t0 = t[1]
        T  = t[end]

        if T - t0 < 2a
            error("Support radius a is too large: need T - t0 >= 2a")
        end

        if K == 1
            return 0.5 * (t0 + T)
        else
            left  = t0 + a
            right = T - a

            # If we want left most center on left, and right most center on right
            # (right - left) / (K - 1) space between two centers

            return left + (k - 1) * (right - left) / (K - 1)
        end
    end


    # ====================================================================================================
    # sine testing functions
    # ====================================================================================================
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


    # ====================================================================================================
    # C infty bump tesing functions
    # ====================================================================================================

    function measure_bump(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int;
        a::Real = 2.0,
        η::Real = 9.0,
    )

        # ??? Should we hardcode a
        a = (t[end] - t[1]) / K + 1

        c = support_center(t, k, K, a)

        function raw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            return exp(-η / h)
        end

        function draw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            f = exp(-η / h)

            # derivative of g(u) = -η / h
            # where h = 1 - (u/a)^2
            g1 = -2.0 * η * u / (a^2 * h^2)

            return f * g1
        end

        function ddraw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            f = exp(-η / h)

            # f'' = f * (g'' + (g')^2), where f = exp(g)
            g1 = -2.0 * η * u / (a^2 * h^2)
            g2 = -2.0 * η / a^2 * (h^(-2) + 4.0 * u^2 / (a^2 * h^3))

            return f * (g2 + g1^2)
        end

        phi_raw = [raw(s - c) for s in t]
        C = l2_scale(t, phi_raw)

        phi   = C .* phi_raw
        dphi  = C .* [draw(s - c) for s in t]
        ddphi = C .* [ddraw(s - c) for s in t]

        return phi, dphi, ddphi
    end


    function measure_bump_function(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int;
        a::Real = 2.0,
        η::Real = 9.0,
    )

        # ??? Should we hardcode a
        a = (t[end] - t[1]) / K + 1

        c = support_center(t, k, K, a)

        function raw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            return exp(-η / h)
        end

        function draw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            f = exp(-η / h)
            g1 = -2.0 * η * u / (a^2 * h^2)

            return f * g1
        end

        function ddraw(u)
            z = u / a
            h = 1.0 - z^2

            if h <= 0.0
                return 0.0
            end

            f = exp(-η / h)

            g1 = -2.0 * η * u / (a^2 * h^2)
            g2 = -2.0 * η / a^2 * (h^(-2) + 4.0 * u^2 / (a^2 * h^3))

            return f * (g2 + g1^2)
        end

        phi_raw = [raw(s - c) for s in t]
        C = l2_scale(t, phi_raw)

        phi(s)   = C * raw(s - c)
        dphi(s)  = C * draw(s - c)
        ddphi(s) = C * ddraw(s - c)

        return phi, dphi, ddphi
    end


    # ====================================================================================================
    # Hartley modulation tesing functions
    # ====================================================================================================

    function measure_hartley(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int;
        a::Real = 0.8,
    )
        # center c_k
        c = support_center(t, k, K, a)

        cas(x)   = cos(x) + sin(x)
        casp(x)  = cos(x) - sin(x)      # derivative of cas
        caspp(x) = -cos(x) - sin(x)     # second derivative of cas

        λ2 = 2π / a
        λ4 = 4π / a
        λ6 = 6π / a

        function raw(u)
            if abs(u) >= a
                return 0.0
            end

            return cas(λ6 * u) -
                   3.0 * cas(λ4 * u) +
                   3.0 * cas(λ2 * u) -
                   1.0
        end

        function draw(u)
            if abs(u) >= a
                return 0.0
            end

            return λ6 * casp(λ6 * u) -
                   3.0 * λ4 * casp(λ4 * u) +
                   3.0 * λ2 * casp(λ2 * u)
        end

        function ddraw(u)
            if abs(u) >= a
                return 0.0
            end

            return λ6^2 * caspp(λ6 * u) -
                   3.0 * λ4^2 * caspp(λ4 * u) +
                   3.0 * λ2^2 * caspp(λ2 * u)
        end

        phi_raw = [raw(s - c) for s in t]
        C = l2_scale(t, phi_raw)

        phi   = C .* phi_raw
        dphi  = C .* [draw(s - c) for s in t]
        ddphi = C .* [ddraw(s - c) for s in t]

        return phi, dphi, ddphi
    end


    function measure_hartley_function(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int;
        a::Real = 0.8,
    )
        # center c_k
        c = support_center(t, k, K, a)

        cas(x)   = cos(x) + sin(x)
        casp(x)  = cos(x) - sin(x)
        caspp(x) = -cos(x) - sin(x)

        λ2 = 2π / a
        λ4 = 4π / a
        λ6 = 6π / a

        function raw(u)
            if abs(u) >= a
                return 0.0
            end

            return cas(λ6 * u) -
                   3.0 * cas(λ4 * u) +
                   3.0 * cas(λ2 * u) -
                   1.0
        end

        function draw(u)
            if abs(u) >= a
                return 0.0
            end

            return λ6 * casp(λ6 * u) -
                   3.0 * λ4 * casp(λ4 * u) +
                   3.0 * λ2 * casp(λ2 * u)
        end

        function ddraw(u)
            if abs(u) >= a
                return 0.0
            end

            return λ6^2 * caspp(λ6 * u) -
                   3.0 * λ4^2 * caspp(λ4 * u) +
                   3.0 * λ2^2 * caspp(λ2 * u)
        end

        # normalization constant
        phi_raw = [raw(s - c) for s in t]
        C = l2_scale(t, phi_raw)

        phi(s)   = C * raw(s - c)
        dphi(s)  = C * draw(s - c)
        ddphi(s) = C * ddraw(s - c)

        return phi, dphi, ddphi
    end


    # ====================================================================================================
    # Polynomial tesing functions
    # ====================================================================================================
    function measure_polynomial(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int,
        m::Int;
        a::Real = 0.52
    )
        # c is the center c_k of this test function
        c = support_center(t, k, K, a)

        # unnormalized φ
        function raw(u)
            if abs(u) >= a
                return 0.0
            end

            return (a^2 - u^2)^m
        end

        # derivative with respect to t
        # since u = t - c, du/dt = 1
        function draw(u)
            if abs(u) >= a
                return 0.0
            end

            return -2.0 * m * u * (a^2 - u^2)^(m - 1)
        end

        # second derivative with respect to t
        function ddraw(u)
            if abs(u) >= a
                return 0.0
            end

            q = a^2 - u^2

            return -2.0 * m * q^(m - 1) +
                   4.0 * m * (m - 1) * u^2 * q^(m - 2)
        end

        # evaluate raw φ on the grid
        phi_raw = [raw(s - c) for s in t]

        # choose C so ∫ φ(t)^2 dt ≈ 1
        C = l2_scale(t, phi_raw)

        phi   = C .* phi_raw
        dphi  = C .* [draw(s - c) for s in t]
        ddphi = C .* [ddraw(s - c) for s in t]

        return phi, dphi, ddphi
    end


    function measure_polynomial_function(
        t::AbstractVector{<:Real},
        k::Int,
        K::Int,
        m::Int;
        a::Real = 0.52
    )
        # center of the kth compactly supported test function
        c = support_center(t, k, K, a)

        # raw polynomial shape
        raw(u) = abs(u) < a ? (a^2 - u^2)^m : 0.0

        # first derivative of raw shape
        draw(u) = abs(u) < a ? -2.0 * m * u * (a^2 - u^2)^(m - 1) : 0.0

        # second derivative of raw shape
        function ddraw(u)
            if abs(u) >= a
                return 0.0
            end

            q = a^2 - u^2

            return -2.0 * m * q^(m - 1) +
                   4.0 * m * (m - 1) * u^2 * q^(m - 2)
        end

        # compute normalization constant C using the grid t
        phi_raw = [raw(s - c) for s in t]
        C = l2_scale(t, phi_raw)

        # actual normalized test functions
        phi(s)   = C * raw(s - c)
        dphi(s)  = C * draw(s - c)
        ddphi(s) = C * ddraw(s - c)

        return phi, dphi, ddphi
    end



    # ====================================================================================================
    # Chebyshev first kind  T_n / \sqrt(1 - x^2)

    # Since Chebyshev coefficients are defined on [-1, 1]
    # We need to transform [t0, tT] to [-1, 1]
    # t in [t0, tT] corresponds to -1 + 2/(tT - t0) * (t - t0)
    # ====================================================================================================

    ### BOUNDARY ISSUE!!  T_n / \sqrt(1 - x^2) at x=1 or -1 is infinite

    function chebyshev_T(t, k)
        t0 = t[1]
        tT = t[end]

        x = (2 .* t .- (t0+tT))/(tT-t0)
        phi = cos.(k .* acos.(x)) ./ sqrt.(1 .- x.^2)

        dxdt = 2/(tT-t0)
        T = cos.(k .* acos.(x))
        Tp = k .* sin.(k .* acos.(x)) ./ sqrt.(1 .- x.^2)

        numerator = Tp .* (1-x.^2) .+ x.*T
        dphidx = numerator ./ (1-x.^2).^(3/2)

        dphi = dphidx .* dxdt

        return phi, dphi
    end


    function chebyshev_T_function(t0, tT, k)
        function phi(t)
            x = (2t-(t0+tT))/(tT-t0)
            return cos(k*acos(x))/sqrt(1-x^2)
        end

        dxdt = 2/(tT-t0)

        function dphi(t)
            x=(2t-(t0+tT))/(tT-t0)
            T = cos(k * acos(x))

            Tp = k*sin(k * acos(x))/sqrt(1-x^2)
            numerator = Tp*(1-x^2) + x*T
            dphidx = numerator/(1-x^2)^(3/2)

            return dphidx*dxdt
        end

        return phi, dphi
    end


    # ====================================================================================================
    # Chebyshev second kind  U_n \sqrt(1 - x^2)
    # ====================================================================================================

    function chebyshev_U(t, k)
        t0 = t[1]
        tT = t[end]

        x = clamp.(
            (2 .* t .- (t0+tT))/(tT-t0),
            -1.0,
            1.0
        )

        # phi = U_k(x)*sqrt(1-x^2)
        # simplified form
        phi = sin.((k+1) .* acos.(x))

        # derivative
        dxdt = 2/(tT-t0)
        dphi = -(k+1) .* cos.((k+1).*acos.(x)) ./ sqrt.(1 .- x.^2)
        dphi = dphi .* dxdt

        return phi, dphi
    end


    function chebyshev_U_function(t, k)
        t0 = t[1]
        tT = t[end]

        function phi(s)
            x = clamp((2s-(t0+tT))/(tT-t0), -1.0, 1.0)
            return sin((k+1)*acos(x))
        end

        function dphi(s)
            dxdt = 2/(tT-t0)
            x = clamp((2s-(t0+tT))/(tT-t0), -1.0, 1.0)

            return -(k+1) *
                   cos((k+1)*acos(x)) /
                   sqrt(1-x^2) *
                   dxdt
        end

        return phi, dphi
    end
end
