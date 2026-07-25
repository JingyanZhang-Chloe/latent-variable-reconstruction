

function interval_integral_poly(qcoeffs::Vector{Float64}, h::Float64)
    # Computes ∫_{-h}^{h} q(z) dz.

    val = 0.0
    deg = length(qcoeffs) - 1

    for n in 0:deg
        c = qcoeffs[n + 1]
        val += c * (h^(n + 1) - (-h)^(n + 1)) / (n + 1)
    end

    return val
end


function trig_moments(za::Float64, zb::Float64, ω::Float64, maxdeg::Int)
    # S[n+1] = ∫ za^zb z^n sin(ωz) dz
    # C[n+1] = ∫ za^zb z^n cos(ωz) dz

    S = zeros(Float64, maxdeg + 1)
    C = zeros(Float64, maxdeg + 1)

    S[1] = (cos(ω * za) - cos(ω * zb)) / ω
    C[1] = (sin(ω * zb) - sin(ω * za)) / ω

    for n in 1:maxdeg
        S[n + 1] =
            (-zb^n * cos(ω * zb) + za^n * cos(ω * za)) / ω +
            (n / ω) * C[n]

        C[n + 1] =
            (zb^n * sin(ω * zb) - za^n * sin(ω * za)) / ω -
            (n / ω) * S[n]
    end

    return S, C
end


function poly_trig_integral(
    coeffs::Vector{Float64},
    a::Float64,
    b::Float64,
    m::Float64,
    ω::Float64,
    t0::Float64;
    trig::Symbol = :sin
)
    # coeffs represent p(z), where z = x - m.
    #
    # Computes either
    # ∫_a^b p(x-m) sin(ω(x-t0)) dx
    #
    # or
    # ∫_a^b p(x-m) cos(ω(x-t0)) dx.

    za = a - m
    zb = b - m

    maxdeg = length(coeffs) - 1
    S, C = trig_moments(za, zb, ω, maxdeg)

    raw_sin = 0.0
    raw_cos = 0.0

    for n in 0:maxdeg
        c = coeffs[n + 1]
        raw_sin += c * S[n + 1]
        raw_cos += c * C[n + 1]
    end

    θ = ω * (m - t0)

    if trig == :sin
        return sin(θ) * raw_cos + cos(θ) * raw_sin

    elseif trig == :cos
        return cos(θ) * raw_cos - sin(θ) * raw_sin

    else
        error("trig must be :sin or :cos")
    end
end


function get_weak_blocks_exact_spline(
    I_data::Vector{Float64},
    t::Vector{Float64},
    K::Int,
    method::String
)
    Y  = zeros(Float64, K)
    W1 = zeros(Float64, K)
    W2 = zeros(Float64, K)
    W3 = zeros(Float64, K)

    Ihat = build_interpolant_exact(t, I_data, method)

    degree = method == "QSpline_exact" ? 2 : 3

    t0 = t[1]
    tT = t[end]
    L = tT - t0

    for k in 1:K
        ω = k * π / L

        # phi_k(x) = sin(ω(x - t0)).
        # Since phi_k(t0) = phi_k(tT) = 0 for this basis,
        # the boundary is usually zero, but keep it for safety.
        phi_t0 = sin(ω * (t0 - t0))
        phi_tT = sin(ω * (tT - t0))

        boundary = phi_tT * Ihat(tT) - phi_t0 * Ihat(t0)

        int_dphi_I = 0.0
        int_phi_I  = 0.0
        int_phi_I2 = 0.0
        int_phi_IF = 0.0

        # F_left stores F(t_i) = ∫_{t0}^{t_i} Ihat(s) ds.
        F_left = 0.0

        for i in 1:(length(t) - 1)
            a = t[i]
            b = t[i + 1]

            qcoeffs, m, h = local_poly_coeffs(Ihat, a, b, degree)

            # Y uses ∫ phi'(x) I(x) dx.
            # phi'(x) = ω cos(ω(x - t0)).
            int_dphi_I += ω * poly_trig_integral(
                qcoeffs,
                a,
                b,
                m,
                ω,
                t0;
                trig = :cos
            )

            # W1 = ∫ phi(x) I(x) dx.
            int_phi_I += poly_trig_integral(
                qcoeffs,
                a,
                b,
                m,
                ω,
                t0;
                trig = :sin
            )

            # W2 = ∫ phi(x) I(x)^2 dx.
            q2coeffs = poly_multiply(qcoeffs, qcoeffs)

            int_phi_I2 += poly_trig_integral(
                q2coeffs,
                a,
                b,
                m,
                ω,
                t0;
                trig = :sin
            )

            # Build local polynomial for F(x) = ∫_{t0}^{x} I(s) ds.
            Fcoeffs = local_F_coeffs(qcoeffs, h, F_left)

            # W3 = ∫ phi(x) I(x) F(x) dx.
            qFcoeffs = poly_multiply(qcoeffs, Fcoeffs)

            int_phi_IF += poly_trig_integral(
                qFcoeffs,
                a,
                b,
                m,
                ω,
                t0;
                trig = :sin
            )

            # Update F_left for the next interval.
            F_left += interval_integral_poly(qcoeffs, h)
        end

        Y[k]  = boundary - int_dphi_I
        W1[k] = int_phi_I
        W2[k] = int_phi_I2
        W3[k] = int_phi_IF
    end

    return Y, W1, W2, W3
end





using BSplineKit

function build_I_and_F_bspline(
    t::AbstractVector,
    I_data::AbstractVector,
    degree::Int
)
    length(t) == length(I_data) ||
        throw(DimensionMismatch(
            "t and I_data must have the same length"
        ))

    issorted(t) ||
        throw(ArgumentError("t must be increasingly sorted"))

    # BSplineKit uses order = degree + 1
    Iinterp = interpolate(
        t,
        I_data,
        BSplineOrder(degree + 1)
    )

    # Extract the underlying B-spline representation
    Ispline = BSplineKit.Splines.spline(Iinterp)

    # Exact analytical antiderivative.
    # By convention F(t[1]) = 0.
    Fspline = BSplineKit.Splines.integral(Ispline)

    return Iinterp, Fspline
end
