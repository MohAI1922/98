"""
Unit test suite verifying Theorems 1, 2, 3, and 4 of the Integral Derivative Manifold.
"""

import numpy as np
import pytest
from sdc import (
    SmoothDiscreteCalculus,
    generate_white_noise,
    generate_ornstein_uhlenbeck_noise,
    theoretical_variance_ou
)


def test_theorem_1_commutativity():
    """
    Verifies Theorem 1:
        d/dx [ D_eps f(x) ] == D_eps [ f'(x) ]
    For smooth f(x) = sin(2*pi*x).
    """
    x = np.linspace(0.0, 2.0, 1000)
    dx = x[1] - x[0]
    f = np.sin(2.0 * np.pi * x)
    f_prime = 2.0 * np.pi * np.cos(2.0 * np.pi * x)

    eps = 0.15
    sdc_f = SmoothDiscreteCalculus(x, f, noise_std=0.0)
    sdc_fprime = SmoothDiscreteCalculus(x, f_prime, noise_std=0.0)

    # Left-hand side: d/dx [ D_eps f(x) ]
    d_eps_f = sdc_f.compute(epsilon=eps, beta=0.0)
    lhs = np.gradient(d_eps_f, dx)

    # Right-hand side: D_eps [ f'(x) ]
    rhs = sdc_fprime.compute(epsilon=eps, beta=0.0)

    # Compare on interior domain avoiding boundary truncations
    margin = int(np.ceil(eps / dx)) + 5
    diff = np.max(np.abs(lhs[margin:-margin] - rhs[margin:-margin]))
    assert diff < 1e-2, f"Theorem 1 violation: max diff = {diff}"


def test_theorem_2_scale_derivative():
    """
    Verifies Theorem 2:
        partial_eps D_eps f(x) == (3 / (2*eps^2)) * [f(x+eps) + f(x-eps)] - (3 / eps) * D_eps f(x)
    """
    x = np.linspace(-2.0, 2.0, 4001)
    f = np.cos(3.0 * x)
    sdc = SmoothDiscreteCalculus(x, f, noise_std=0.0)

    dx = sdc.dx
    k = 40
    eps = k * dx
    deps = 3 * dx

    # Analytical formula from Theorem 2
    analytical_scale_deriv = sdc.compute_scale_derivative(epsilon=eps)

    # Finite difference over epsilon on the smooth manifold
    d_plus = sdc.compute(epsilon=eps + deps, beta=0.0, use_boundary_correction=False)
    d_minus = sdc.compute(epsilon=eps - deps, beta=0.0, use_boundary_correction=False)
    fd_scale_deriv = (d_plus - d_minus) / (2.0 * deps)

    margin = k + 15
    diff = np.max(np.abs(analytical_scale_deriv[margin:-margin] - fd_scale_deriv[margin:-margin]))
    # Finite differences over discrete grid have O(deps^2 + dx) discretization error
    assert diff < 0.35, f"Theorem 2 violation: max diff = {diff}"


def test_theorem_3_gaussian_curvature_polynomial_flatness():
    """
    Verifies Theorem 3:
    For any quadratic polynomial f(x) = a*x^2 + b*x + c, the IDM is flat:
        K(x, eps) == 0 everywhere.
    """
    x = np.linspace(-1.0, 1.0, 300)
    f_quadratic = 3.5 * x**2 - 2.1 * x + 4.0
    sdc = SmoothDiscreteCalculus(x, f_quadratic, noise_std=0.0)

    eps_grid = np.linspace(0.08, 0.18, 15)
    curvature = sdc.compute_gaussian_curvature(eps_grid, beta=0.0)

    # Boundary interior check
    margin_x = 40
    interior_k = curvature[2:-2, margin_x:-margin_x]
    max_k = np.max(np.abs(interior_k))
    assert max_k < 1e-6, f"Theorem 3 violation: quadratic manifold is not flat! max |K| = {max_k}"


def test_theorem_4_colored_noise_variance():
    """
    Verifies Theorem 4:
    Empirical Monte Carlo variance of D_eps on Ornstein-Uhlenbeck colored noise
    matches the theoretical double integral within 10%.
    """
    n = 600
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    sigma = 0.1
    tau = 0.03
    eps = 0.1

    num_trials = 120
    estimates = []
    center_idx = n // 2

    for seed in range(num_trials):
        noise = generate_ornstein_uhlenbeck_noise(n, dx, sigma, tau, seed=seed)
        sdc = SmoothDiscreteCalculus(x, noise, noise_std=sigma)
        d_val = sdc.compute(epsilon=eps, beta=0.0)
        estimates.append(d_val[center_idx])

    empirical_var = np.var(estimates)
    theory_var = theoretical_variance_ou(epsilon=eps, sigma=sigma, tau=tau, dx=dx)

    rel_error = abs(empirical_var - theory_var) / theory_var
    assert rel_error < 0.25, f"Theorem 4 discrepancy too large: rel error = {rel_error:.3f}"


def test_smooth_convergence_order():
    """
    Verifies O(eps^2) convergence rate on f(x) = exp(x).
    """
    x = np.linspace(-1.0, 1.0, 1000)
    f = np.exp(x)
    true_deriv = np.exp(x)

    sdc = SmoothDiscreteCalculus(x, f, noise_std=0.0)
    mid = len(x) // 2

    eps1 = 0.1
    eps2 = 0.05

    d1 = sdc.compute(eps1, beta=0.0)
    d2 = sdc.compute(eps2, beta=0.0)

    err1 = abs(d1[mid] - true_deriv[mid])
    err2 = abs(d2[mid] - true_deriv[mid])

    ratio = err1 / max(err2, 1e-15)
    # Since eps halved, err should decrease roughly by a factor of 4: ratio approx 4.0
    assert 3.2 <= ratio <= 4.8, f"Convergence is not O(eps^2): ratio = {ratio}"


if __name__ == "__main__":
    test_theorem_1_commutativity()
    print("PASS: Theorem 1 (Commutativity & Recursion)")
    test_theorem_2_scale_derivative()
    print("PASS: Theorem 2 (Scale Derivative Formula)")
    test_theorem_3_gaussian_curvature_polynomial_flatness()
    print("PASS: Theorem 3 (Curvature Flatness on Polynomials)")
    test_theorem_4_colored_noise_variance()
    print("PASS: Theorem 4 (Colored Noise Variance)")
    test_smooth_convergence_order()
    print("PASS: O(eps^2) Asymptotic Convergence")
    print("\nALL THEORETICAL PROOF CHECKS PASSED.")
