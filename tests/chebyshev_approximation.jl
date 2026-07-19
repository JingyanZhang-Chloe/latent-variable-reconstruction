# chebyshev_approximation.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 14/07/2026
=#

"""
We have, after mapping t to x in [-1, 1]

Ihat(t) = ∑ a_k U_{k-1} (x(t))
where x_from_t(t, t0, tT) = (2t - (t0 + tT)) / (tT - t0)
"""


x_from_t(t, t0, tT) = (2t - (t0 + tT)) / (tT - t0)


function chebyshev_coefficients(Ihat, t, K)
    """
    Since change of variable, a_k = 4 / πL W1_k

    a[k] is the coefficient multiplying U_{k-1}
    """
    a = zeros(K)
    t0 = t[1]
    tT = t[end]
    L = tT - t0

    for k in 1:K
        phi, dphi = Measure.chebyshev_U_function(t, k)

        W1_k = quadgk(x -> phi(x) * Ihat(x), t)[1]
        a[k] = 4.0 / (π * L) * W1_k
    end

    return a
end


function chebyshev_U(n::Int, x::Real)
    x = clamp(x, -1.0, 1.0)

    n == 0 && return 1.0
    n == 1 && return 2.0 * x

    U_nm2 = 1.0
    U_nm1 = 2.0 * x

    for j in 2:n
        U_n = 2.0 * x * U_nm1 - U_nm2
        U_nm2, U_nm1 = U_nm1, U_n
    end

    return U_nm1
end


function I_chebyshev_U(s, t, coeffs)
    t0 = t[1]
    tT = t[end]

    x = clamp(
        (2s - (t0 + tT)) / (tT - t0),
        -1.0,
        1.0
    )

    return sum(
        coeffs[k] * chebyshev_U(k - 1, x)
        for k in eachindex(coeffs)
    )
end



using Plots

function plot_chebyshev_U_approximation(Ihat, t, K; together=true)
    coeffs = chebyshev_coefficients(Ihat, t, K)

    t_dense = collect(
        range(t[1], t[end], length=2000)
    )

    I_spline = Ihat.(t_dense)

    I_cheb = [
        I_chebyshev_U(s, t, coeffs)
        for s in t_dense
    ]

    mask = (t_dense .> 500) .& (t_dense .<= 1000)
    t_dense = t_dense[mask]
    I_cheb = I_cheb[mask]
    I_spline = I_spline[mask]

    if together
        plt = plot(
            t_dense,
            I_cheb;
            linewidth=2.5,
            linealpha=1.0,
            label="Chebyshev-U approximation",
        )

        plot!(
            plt,
            t_dense,
            I_spline;
            linewidth=15,
            linealpha=0.1,
            label="Ihat",
            xlabel="t",
            ylabel="I(t)",
            title="Chebyshev-U approximation, K = $K",
        )

        return plt
    else
        plt1 = plot(
            t_dense,
            I_spline;
            linewidth=3,
            label="Ihat",
            xlabel="t",
            ylabel="I(t)",
            title="Ihat"
        )

        plt2 = plot(
            t_dense,
            I_cheb;
            linewidth=3,
            label="Icheb",
            xlabel="t",
            ylabel="I(t)",
            title="Chebyshev-U approximation with K = $K"
        )

        display(plt1)
        display(plt2)
    end
end


function main()
    t = collect(0.0:10.0:1000.0)
    S, I, R = Logic.simulate_sir(t)

    noise = 0
    I_data = I .+ noise .* I .* randn(length(I))

    # Optional: avoid negative infected values after adding noise
    I_data = max.(I_data, 0.0)

    K = 30
    Ihat = AkimaInterpolation(I_data, t)
    plt = plot_chebyshev_U_approximation(Ihat, t, K; together=true)
    display(plt)
end

main()
