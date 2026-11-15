"""
Stochastic noise models and analytical variance functions for IDM.
Supports both White Gaussian Noise (AWGN) and Ornstein-Uhlenbeck (Colored) Noise.
"""

from __future__ import annotations
import numpy as np


def generate_white_noise(n: int, sigma: float, seed: int | None = None) -> np.ndarray:
    """Generates zero-mean white Gaussian noise with standard deviation sigma."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=n)


def generate_ornstein_uhlenbeck_noise(
    n: int,
    dx: float,
    sigma: float,
    correlation_length: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Generates stationary Ornstein-Uhlenbeck colored noise:
        d eta = - (1 / tau) * eta * dt + sqrt(2 * sigma^2 / tau) * dW
    where tau = correlation_length.
    The covariance satisfies: E[eta(x) eta(x+h)] = sigma^2 * exp(-|h| / tau).
    """
    rng = np.random.default_rng(seed)
    tau = max(1e-12, correlation_length)
    decay = np.exp(-dx / tau)
    noise_variance = sigma**2 * (1.0 - decay**2)
    noise_std = np.sqrt(max(0.0, noise_variance))

    eta = np.zeros(n, dtype=np.float64)
    eta[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        eta[i] = eta[i - 1] * decay + rng.normal(0.0, noise_std)
    return eta


def theoretical_variance_ou(
    epsilon: float,
    sigma: float,
    tau: float,
    dx: float
) -> float:
    """
    Computes numerical/discrete variance of the IDM operator (beta=0) under OU noise:
    Var(D_eps f) = sum_i sum_j K(t_i) K(t_j) Cov(eta_i, eta_j) * dx^2
    """
    k_len = max(1, int(np.ceil(epsilon / dx)))
    t = np.arange(-k_len, k_len + 1) * dx
    kernel = (3.0 / (2.0 * (epsilon**3))) * t * dx
    cov_matrix = sigma**2 * np.exp(-np.abs(np.subtract.outer(t, t)) / max(1e-12, tau))
    return float(kernel @ cov_matrix @ kernel)
