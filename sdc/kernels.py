"""
Parametric and Boundary-Adapted Moment Kernels for Integral Derivative Manifolds.
"""

from __future__ import annotations
import numpy as np
from scipy.special import beta as beta_func


def get_normalization_constant(beta: float) -> float:
    """
    Computes C_beta = B(3/2, beta + 1).
    For beta = 0: C_0 = 2/3
    For beta = 1: C_1 = 4/15
    For beta = 2: C_2 = 16/105
    """
    if beta == 0.0:
        return 2.0 / 3.0
    elif beta == 1.0:
        return 4.0 / 15.0
    elif beta == 2.0:
        return 16.0 / 105.0
    else:
        return float(beta_func(1.5, beta + 1.0))


def compute_kernel(
    t: np.ndarray,
    epsilon: float,
    beta: float = 0.0,
    dx: float | None = None
) -> np.ndarray:
    """
    Evaluates the parametric moment kernel:
        K_eps(t) = (1 / (C_beta * eps^3)) * t * (1 - (t/eps)^2)^beta  for |t| <= eps
    If dx is provided, normalizes kernel so that discrete Riemann sum:
        sum_i t_i * K(t_i) * dx == 1.0
    """
    c_beta = get_normalization_constant(beta)
    u = t / epsilon
    mask = np.abs(u) <= 1.0
    kernel = np.zeros_like(t, dtype=np.float64)
    if beta == 0.0:
        kernel[mask] = (1.0 / (c_beta * (epsilon**3))) * t[mask]
    else:
        kernel[mask] = (1.0 / (c_beta * (epsilon**3))) * t[mask] * ((1.0 - u[mask]**2) ** beta)
    
    if dx is not None:
        moment = np.sum(kernel[mask] * t[mask]) * dx
        if abs(moment) > 1e-12:
            kernel = kernel / moment
    return kernel


def compute_boundary_kernel(
    t: np.ndarray,
    epsilon: float,
    c_left: float,
    c_right: float
) -> np.ndarray:
    """
    Computes boundary-corrected linear moment kernel for truncated intervals [-c_left * eps, c_right * eps].
    Ensures O(eps^2) bias preservation at edges.
    """
    u = t / epsilon
    mask = (u >= -c_left) & (u <= c_right)
    # Fit affine coefficients [a0, a1] such that moments match:
    # int_{-c_left}^{c_right} (a0 + a1*u) du = 0
    # int_{-c_left}^{c_right} u * (a0 + a1*u) du = 1
    m0 = c_right + c_left
    m1 = 0.5 * (c_right**2 - c_left**2)
    m2 = (1.0 / 3.0) * (c_right**3 + c_left**3)
    
    # [ [m0, m1], [m1, m2] ] [a0, a1]^T = [0, 1]^T
    det = m0 * m2 - m1**2
    if np.abs(det) < 1e-12:
        return np.zeros_like(t)
    
    a0 = -m1 / det
    a1 = m0 / det
    
    kernel = np.zeros_like(t, dtype=np.float64)
    kernel[mask] = (1.0 / (epsilon**2)) * (a0 + a1 * u[mask])
    return kernel
