# Latent Variable Reconstruction via Weak-Form Integral Elimination & HC-LS

[![Julia Version](https://img.shields.io/badge/julia-v1.9%2B-blue.svg)](https://julialang.org/)
[![Base Thesis](https://img.shields.io/badge/thesis-model--integration-orange.svg)](https://github.com/JingyanZhang-Chloe/model-integration)

An advanced Julia framework for state reconstruction and global parameter estimation in partially observed non-linear dynamical systems (e.g., SIR/SEIR compartmental models) under observational noise. 

This repository extends the theoretical groundwork of [model-integration](https://github.com/JingyanZhang-Chloe/model-integration) by introducing weak-form variational quadrature, custom spline antiderivative solvers, automatic log-dispersion time-scale optimization, and a unified **Homotopy Continuation–Least Squares (HC-LS)** optimization pipeline.

---

## Technical Overview

Parameter estimation in partially observed ODE systems $\dot{x}(t) = f(x(t), p)$ with unobserved latent states (e.g., observing $I(t)$ while $S(t)$ and $E(t)$ remain unmeasured) faces severe structural challenges:

1. **Differential Elimination Latency & Noise Amplification:** Standard differential elimination requires high-order derivatives $I'(t), I''(t), I'''(t)$, which exponentially amplify high-frequency measurement noise.
2. **Local Minima in Non-Convex Optimization:** Direct non-linear least-squares (NLS) on raw ODE solvers depends heavily on initial guesses $p_0$ and easily traps in local minima along non-identifiable parameter ridges.

### The Weak-Form Integral Elimination Solution
By integrating the governing ODE system, unobserved compartments are eliminated algebraically to yield a weak polynomial identity tested against smooth basis functions $\{\phi_k\}_{k=1}^K$:

$$L_k = \sum_{j=1}^m c_j(p) B_{j,k}, \quad k = 1, \dots, K$$

For the **SEIR model**, testing the eliminated equation yields:

$$L_k = \sigma(E_0 + I_0) B_{0,k} - (\gamma + \sigma) B_{1,k} - \frac{\alpha}{2}\left(B_{2,k} - I_0^2 B_{0,k}\right) + \left[\alpha\sigma(S_0 + E_0 + I_0) - \sigma\gamma\right] B_{3,k} - \alpha(\gamma + \sigma) B_{4,k} - \frac{\alpha\sigma\gamma}{2} B_{5,k}$$

Where the boundary-eliminated left-hand block $L_k$ and weighted integral blocks $B_{j,k}$ are defined by:
* $L_k = [I(t)\phi_k(t)]_0^T - \int_0^T I(t)\phi_k'(t)\,dt$
* $B_{0,k} = \int_0^T \phi_k(t)\,dt$
* $B_{1,k} = \int_0^T I(t)\phi_k(t)\,dt$
* $B_{2,k} = \int_0^T I(t)^2\phi_k(t)\,dt$
* $B_{3,k} = \int_0^T F(t)\phi_k(t)\,dt, \quad \text{where } F(t) = \int_0^t I(s)\,ds$
* $B_{4,k} = \int_0^T G(t)\phi_k(t)\,dt, \quad \text{where } G(t) = \int_0^t I(s)^2\,ds$
* $B_{5,k} = \int_0^T F(t)^2\phi_k(t)\,dt$

Integration by parts transfers derivatives onto the test functions $\phi_k(t)$, completely eliminating numerical differentiation of noisy data.

---

## Testing Functions

### Chebyshev Polynomial Basis
Chebyshev polynomials of the second kind mapped to $[t_0, t_T]$ via $x(t) = \frac{2t - (t_0 + t_T)}{t_T - t_0}$ serve as the default test functions:

$$\phi_k(t) = \sin\left(k \arccos(x(t))\right), \quad k = 1, \dots, K$$

The derivative $\phi_k'(t) = -\frac{2k}{t_T - t_0} \frac{\cos(k \arccos(x(t)))}{\sqrt{1 - x(t)^2}}$ contains a singular factor $1/\sqrt{1 - x(t)^2}$ at the endpoints $x = \pm 1$.

**Singularity Cancellation Transformation:** Applying $x = \cos\theta$ ($\theta \in [0, \pi]$) introduces Jacobian $\frac{dt}{d\theta} = -\frac{t_T - t_0}{2}\sin\theta$. The singular factor cancels analytically, yielding a smooth and well-conditioned integral:

$$L_k = k \int_0^\pi I(t(\theta)) \cos(k\theta) \, d\theta$$

### Supported Test Function Families
* **Chebyshev Basis (Default):** Orthogonal, vanishing boundaries, nested frequency decay.
* **Trigonometric Sines:** Harmonic basis $\phi_k(t) = \sin\left(\frac{k\pi t}{T}\right)$.
* **$C^\infty$ Bump Functions:** Compactly supported smooth functions $\phi(t; a, c_k) = C \exp\left(-\frac{\eta}{1 - ((t-c_k)/a)^2}\right)$.
* **Hartley Modulating Functions:** Shifted Hartley kernels $cas(t) = \cos(t) + \sin(t)$.
* **Polynomial Test Functions:** High-order compact polynomials $\phi_k(t; a, c_k) = C (a^2 - (t - c_k)^2)^m$.

---

## Data Interpolation & Quadrature Schemes

To accommodate noisy or irregularly sampled time series, multiple integration and curve-fitting schemes are provided:

### 1. Direct Grid Quadrature
* **`"T"`:** Composite cumulative trapezoidal rule.
* **`"S"`, `"S_uniform"`:** Cartwright non-uniform and uniform Simpson’s 1/3 rules.
* **`"S_improved"`:** Local 2nd-degree Lagrange interpolation integrated via `QuadGK.jl`.
* **`"S_formula_improved"`:** Exact closed-form antiderivatives for trigonometric test functions.

### 2. Spline & Smoothing Methods
* **`QSpline` / `CubicSpline` / `AkimaInterpolation`:** $C^1/C^2$ exact interpolating splines.
* **`BSpline`:** Flexible B-spline interpolation (orders 2–6) with exact antiderivative evaluation.
* **`BSplineApprox`:** Regression B-spline smoothing with $h = \max(5, \lfloor 0.3 M \rfloor)$ control points to filter observational noise.
* **`RegularizationSmooth`:** Penalized least-squares smoothing spline minimizing $\sum (f(t_i) - I_i)^2 + \lambda \int (f^{(d)}(t))^2 dt$.
* **`PCHIP`:** Shape-preserving, non-oscillatory piecewise cubic Hermite interpolation.

---

## Automated Hyperparameter Selection

### 1. Automated System Size $K$
The number of test functions $K$ is selected dynamically based on weak block decay. The algorithm computes block magnitudes $d_k = \max(\vert{}L_k\vert{}, \vert{}B_{0,k}\vert{}, \dots, \vert{}B_{q,k}\vert{})$ and chooses the smallest $K$ where $d_k \le \tau$ ($\tau = 10^{-2}$) for $r = 3$ consecutive indices.

### 2. Automated Time Rescaling $T$
To prevent matrix ill-conditioning from higher-order double integrals ($B_{4,k}, B_{5,k}$ scaling as $\mathcal{O}(T^3)$), time is rescaled as $t_{\text{scaled}} = t / T$. $T^* = 10^{q^*}$ is selected by minimizing the variance of log-transformed scaled block norms:

$$q^* = \arg\min_{q \in [q_{\text{min}}, q_{\text{max}}]} \text{Var}\left( \left\lbrace \log_{10} \left( \frac{\|B_j\|}{(10^q)^{a_j}} + \epsilon \right) \right\rbrace_j \right)$$

---

## Conclusion & Key Results

This repository demonstrates that **weak-form integral elimination** offers a highly stable, initial-guess-free alternative to traditional differential elimination for epidemic modeling. 

Based on our extensive benchmarking and real-world validations, the Homotopy Continuation-Least Squares (HC-LS) framework achieves the following:
* **Robust Global Recovery:** Accurately recovers latent variables and parameters without requiring prior initial guesses, overcoming the local minima traps of standard optimization.
* **Noise Resilience:** Achieves near machine-precision accuracy in low-noise regimes and maintains robust estimates under moderate-to-high noise (1% - 20%) by leveraging regression B-splines and regularized smoothing as variational filters.
* **Automated Stability:** Dynamically auto-tunes the system size ($K$) and time-rescaling factor ($T$) to prevent matrix ill-conditioning, ensuring mathematical stability across diverse epidemic regimes.
* **Real-World Validation:** Successfully reconstructs some historical epidemic dynamics, including the 1967 Tristan da Cunha common cold outbreak and the 1665 Eyam plague, effectively handling sparse and noisy reporting data.
