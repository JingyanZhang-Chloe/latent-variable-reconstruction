# latent-variable-reconstruction
This repository continues and extends the Bachelor Thesis project originally developed in:  [model-integration](https://github.com/JingyanZhang-Chloe/model-integration). The goal of this version is to clean the implementation, improve the algorithm, and prepare reproducible experiments.

## Different methods of Integration for Weak Form

- `"T"`: Trapezoidal Integration method, corresponding to `cumul_integrate(x, y)` from `Integrate`
- `"S"`: Simpson's Integration method, corresponding to `cumintegrate(x, y)` implemented in `Integrate.jl`
- `"S_uniform"`: Based on Simpson's method, but uniform `cumintegrate_simpson_uniform(x, y)`
- `"S_improved"`: `cumintegrate_improved(x, y, measure, rtol)`
giving a measure function, compute $$\int_x y * measure\,dx$$ We only approximate y using Lagrange interpolation formula 
$$y \approx L = y0*l0 + y1*l1 + y2*l2$$
and compute the integral using `quadgk`
    
- `"S_formula_improved"` (hardcode sine/cosine testing functions): `cumintegrate_formula_improved(x, y, k, t, basis, derivative)` same idea as `"S_improved"`, but instead of using `quadgk`, use the anti-derivative formula to compute the integral.
  - `basis`: `:sin` or `:cos` (corresponding to testing function sine or cosine)
  - `derivative`: `false` or `true` (corresponding to phi or dphi)
  - Math can be found in section 2.4.2 Improved Formula based Simpson

The following methods are not through function `integrate` in `Integrate.jl`

- `"QSpline_GK"` `"CSpline_GK"` `"Akima_GK"`: using functions in package `DataInterpolations`
  (`QuadraticSpline` `CubicSpline` `AkimaInterpolation`) to approximate C1 or C2 of y. Then use `quadgk` to compute integrals

- `"BSpline_GK"`:

- [TODO] `"Qspline_exact"` `"CSpine_exact"` `"Akima_exact"` (hardcode sine/cosine testing functions): using functions in package `DataInterpolations`
  (`QuadraticSpline` `CubicSpline` `AkimaInterpolation`) to approximate $\hat{y}$ (C1 or C2) of y. Then write $$\hat{y}(x) = c0 + c1*z + c2*z^2 + c3*z^3$$ as polynomial. 
Then use anti-derivative formula to compute integrals.

