# wake_form.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 19/06/2026
=#


include("SEIRModels.jl")
include("../Measure.jl")
using .Value
using .Logic
using .Integrate
using .Measure
using HomotopyContinuation


function get_wake_blocks(I_data::Vector{Float64}, t::Vector{Float64}, method::String, K::Int)
    I0 = I_data[1]

    # Since we are differentiating, maybe it is better to write in the orginal form
    Y  = zeros(K)   # weak left-hand side ∫ phi I'
    W1 = zeros(K)   # ∫ phi
    W2 = zeros(K)   # ∫ phi I
    W3 = zeros(K)   # ∫ phi (I^2 - I0^2)
    W4 = zeros(K)   # ∫ phi F
    W5 = zeros(K)   # ∫ phi G
    W6 = zeros(K)   # ∫ phi F^2
    # Using integration by parts: ∫ phi I' = [phi I]_0^T - ∫ phi' I
    # If we ensure [phi I]_0^T = 0 we have LHS: - ∫ phi' I

    F = Integrate.integrate(t, I_data, method)
    G = Integrate.integrate(t, I_data.^2, method)

    for k in 1:K
        phi, dphi = Measure.measure_sine(t, k)

        Y[k]  = - Integrate.integrate(t, dphi .* I_data, method)[end]

        W1[k] = Integrate.integrate(t, phi)[end]
        W2[k] = Integrate.integrate(t, phi .* I_data)[end]
        W3[k] = Integrate.integrate(t, phi .* (I_data.^2 .- I0^2))[end]
        W4[k] = Integrate.integrate(t, phi .* F)[end]
        W5[k] = Integrate.integrate(t, phi .* G)[end]
        W6[k] = Integrate.integrate(t, phi .* (F.^2))[end]
    end

    return Y, W1, W2, W3, W4, W5, W6
end


function weak_I_hat(paras, I0, W1, W2, W3, W4, W5, W6)
    """
    Vector of length K, containing the RHS of the equation computed by the formula
    When computing residual, instead, we do weak_I_hat - Y
    """
    α_eff  = paras[1]
    σ_eff  = paras[2]
    γ_eff  = paras[3]
    S0_eff = paras[4]
    E0_eff = paras[5]

    C1 = σ_eff * (E0_eff + I0) .* t .* W1
    C2 = - (γ_eff + σ_eff) .* W2
    C3 = - 0.5 * α_eff .* W3
    C4 = (α_eff * σ_eff * (S0_eff + E0_eff + I0) - σ_eff * γ_eff) .* W4
    C5 = - α_eff * (γ_eff + σ_eff) .* W5
    C6 = - 0.5 * α_eff * σ_eff * γ_eff .* W6

    # Here we dont have I0 anymore? (since we do differentiate and then integrate)
    return C1 .+ C2 .+ C3 .+ C4 .+ C5 .+ C6
end


function main()
    println("Hello, Julia!")
end

main()
