"""
Comprehensive Benchmarking Suite for Integral Derivative Manifolds (IDM)
Evaluates IDM against:
  1. Central Finite Differences
  2. Fixed-Scale Lanczos Differentiation (beta = 0)
  3. Fixed-Scale Epanechnikov IDM (beta = 1)
  4. Savitzky-Golay Filter (Degree 2)
  5. Cubic Smoothing Splines (UnivariateSpline with GCV/variance parameter)
  6. Tikhonov Regularized Differentiation (Second-Order Difference Penalty)
  7. IDM Scale-Adaptive (Two-Pass with Epanechnikov kernel)
Across both White and Ornstein-Uhlenbeck (Colored) Noise Regimes.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from sdc import (
    SmoothDiscreteCalculus,
    generate_white_noise,
    generate_ornstein_uhlenbeck_noise
)


def rmse(pred: np.ndarray, truth: np.ndarray, margin: int = 25) -> float:
    """Computes Root Mean Squared Error over the interior domain."""
    p = pred[margin:-margin]
    t = truth[margin:-margin]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def compute_tikhonov_derivative(x: np.ndarray, y: np.ndarray, lam: float = 1e5) -> np.ndarray:
    """Computes derivative via second-order Tikhonov regularization (Whittaker-Eilers)."""
    n = len(x)
    dx = x[1] - x[0]
    diags = [1.0, -2.0, 1.0]
    d2 = sparse.diags(diags, [0, 1, 2], shape=(n - 2, n), dtype=float)
    a_mat = sparse.eye(n, dtype=float) + lam * (d2.T @ d2)
    y_smooth = spsolve(a_mat.tocsc(), y)
    return np.gradient(y_smooth, dx)


def generate_lorenz_series(n_points: int = 2500, dt: float = 0.004) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulates the chaotic Lorenz attractor and returns time, x-trajectory, and dx/dt."""
    def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_span = (0.0, n_points * dt)
    t_eval = np.linspace(0.0, n_points * dt, n_points)
    sol = solve_ivp(lorenz, t_span, [1.0, 1.0, 1.0], t_eval=t_eval, rtol=1e-10, atol=1e-10)
    x_traj = sol.y[0]
    dx_true = 10.0 * (sol.y[1] - sol.y[0])  # dx/dt = sigma * (y - x)
    return t_eval, x_traj, dx_true


def run_chirp_benchmark():
    print("\n" + "=" * 90)
    print("BENCHMARK 1: Nonlinear Chirp Signal f(x) = sin(x^2), x in [0.5, 4.0]")
    print("=" * 90)
    x = np.linspace(0.5, 4.0, 2000)
    dx = x[1] - x[0]
    f_clean = np.sin(x**2)
    f_prime_true = 2.0 * x * np.cos(x**2)

    noise_levels = [0.02, 0.10, 0.25]
    header = f"{'Noise (sigma)':<14} | {'Finite Diff':<11} | {'Lanczos':<10} | {'IDM (b=1)':<10} | {'Savitzky-G':<11} | {'Cubic Spline':<12} | {'Tikhonov':<10} | {'IDM Adaptive':<12}"
    print(header)
    print("-" * len(header))

    for sigma in noise_levels:
        noise = generate_white_noise(len(x), sigma, seed=42)
        y_noisy = f_clean + noise

        # 1. Central Finite Difference
        d_fd = np.gradient(y_noisy, dx)

        # 2. Fixed Lanczos (beta = 0, eps = 0.08)
        sdc = SmoothDiscreteCalculus(x, y_noisy, noise_std=sigma)
        d_lanczos = sdc.compute(epsilon=0.08, beta=0.0)

        # 3. Fixed Epanechnikov IDM (beta = 1, eps = 0.08)
        d_idm_fix = sdc.compute(epsilon=0.08, beta=1.0)

        # 4. Savitzky-Golay (Degree 2)
        w_len = max(5, int(0.12 / dx) | 1)
        d_sg = savgol_filter(y_noisy, window_length=w_len, polyorder=2, deriv=1, delta=dx)

        # 5. Cubic Smoothing Spline
        spl = UnivariateSpline(x, y_noisy, s=len(x) * (sigma ** 2))
        d_spl = spl.derivative(n=1)(x)

        # 6. Tikhonov Regularization
        # Optimal lambda scales as sigma^2 / dx^4
        lam_val = 5e4 * (sigma / 0.10)
        d_tik = compute_tikhonov_derivative(x, y_noisy, lam=lam_val)

        # 7. IDM Scale-Adaptive (Two-pass Epanechnikov)
        d_idm_adapt = sdc.compute_adaptive(beta=1.0, noise_type="white")

        err_fd = rmse(d_fd, f_prime_true)
        err_lz = rmse(d_lanczos, f_prime_true)
        err_idm_fix = rmse(d_idm_fix, f_prime_true)
        err_sg = rmse(d_sg, f_prime_true)
        err_spl = rmse(d_spl, f_prime_true)
        err_tik = rmse(d_tik, f_prime_true)
        err_idm_adapt = rmse(d_idm_adapt, f_prime_true)

        print(f"AWGN s={sigma:<5.2f}  | {err_fd:<11.4f} | {err_lz:<10.4f} | {err_idm_fix:<10.4f} | {err_sg:<11.4f} | {err_spl:<12.4f} | {err_tik:<10.4f} | {err_idm_adapt:<12.4f}")


def run_colored_noise_lorenz_benchmark():
    print("\n" + "=" * 90)
    print("BENCHMARK 2: Chaotic Lorenz Attractor with Colored (OU) Noise (sigma = 0.50)")
    print("=" * 90)
    t, x_traj, dx_true = generate_lorenz_series(n_points=2500, dt=0.004)
    dt = t[1] - t[0]

    sigma = 0.50
    tau_levels = [0.01, 0.04, 0.10]
    header = f"{'Correlation tau':<15} | {'Finite Diff':<11} | {'Savitzky-G':<11} | {'Cubic Spline':<12} | {'Tikhonov':<10} | {'IDM (b=1, fix)':<14} | {'IDM OU-Adapt':<12}"
    print(header)
    print("-" * len(header))

    for tau in tau_levels:
        noise = generate_ornstein_uhlenbeck_noise(len(t), dt, sigma=sigma, correlation_length=tau, seed=123)
        y_noisy = x_traj + noise

        sdc = SmoothDiscreteCalculus(t, y_noisy, noise_std=sigma)

        # 1. Finite Difference
        d_fd = np.gradient(y_noisy, dt)

        # 2. Savitzky-Golay
        w_len = max(5, int(0.12 / dt) | 1)
        d_sg = savgol_filter(y_noisy, window_length=w_len, polyorder=2, deriv=1, delta=dt)

        # 3. Cubic Spline
        spl = UnivariateSpline(t, y_noisy, s=len(t) * (sigma ** 2))
        d_spl = spl.derivative(n=1)(t)

        # 4. Tikhonov
        d_tik = compute_tikhonov_derivative(t, y_noisy, lam=5e4)

        # 5. IDM Fixed Epanechnikov (beta = 1, eps = 0.08)
        d_idm_fix = sdc.compute(epsilon=0.08, beta=1.0)

        # 6. IDM OU-Adaptive (Theorem 5)
        d_idm_ou = sdc.compute_adaptive(beta=1.0, noise_type="colored", tau=tau)

        err_fd = rmse(d_fd, dx_true)
        err_sg = rmse(d_sg, dx_true)
        err_spl = rmse(d_spl, dx_true)
        err_tik = rmse(d_tik, dx_true)
        err_fix = rmse(d_idm_fix, dx_true)
        err_ou = rmse(d_idm_ou, dx_true)

        print(f"tau={tau:<11.3f} | {err_fd:<11.4f} | {err_sg:<11.4f} | {err_spl:<12.4f} | {err_tik:<10.4f} | {err_fix:<14.4f} | {err_ou:<12.4f}")


def generate_benchmark_plot():
    """Generates comparative visualization of IDM vs baselines."""
    x = np.linspace(0.5, 3.5, 1200)
    dx = x[1] - x[0]
    f_clean = np.sin(x**2)
    f_prime = 2.0 * x * np.cos(x**2)
    sigma = 0.08
    y_noisy = f_clean + generate_white_noise(len(x), sigma, seed=99)

    sdc = SmoothDiscreteCalculus(x, y_noisy, noise_std=sigma)
    d_fd = np.gradient(y_noisy, dx)
    d_sg = savgol_filter(y_noisy, window_length=25, polyorder=2, deriv=1, delta=dx)
    d_idm = sdc.compute_adaptive(beta=1.0)

    spl = UnivariateSpline(x, y_noisy, s=len(x) * (sigma ** 2))
    d_spl = spl.derivative(n=1)(x)

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs[0].plot(x, f_clean, 'k-', label='True Signal f(x)', lw=1.5)
    axs[0].plot(x, y_noisy, 'c.', alpha=0.3, label=f'Noisy Observation (sigma={sigma})')
    axs[0].set_ylabel('f(x)')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Signal & Derivative Reconstruction Comparison')

    axs[1].plot(x, f_prime, 'k-', label="Analytical Ground Truth f'(x)", lw=1.5)
    axs[1].plot(x, d_fd, 'r--', alpha=0.35, label='Central Finite Difference (Explodes)')
    axs[1].plot(x, d_sg, 'b--', alpha=0.7, label='Savitzky-Golay (Degree 2)')
    axs[1].plot(x, d_spl, 'orange', alpha=0.7, label='Cubic Smoothing Spline')
    axs[1].plot(x, d_idm, 'g-', lw=2.0, label='IDM Epanechnikov Scale-Adaptive (Proposed)')
    axs[1].set_ylim([-9.0, 9.0])
    axs[1].set_ylabel("f'(x)")
    axs[1].legend(loc='lower left')

    # Curvature profile K(x, eps)
    eps_grid = np.linspace(0.04, 0.16, 20)
    K = sdc.compute_gaussian_curvature(eps_grid, beta=1.0)
    mean_k = np.mean(np.abs(K), axis=0)
    axs[2].plot(x, mean_k, 'm-', lw=1.5, label='IDM Mean |Gaussian Curvature| |K(x, eps)|')
    axs[2].set_ylabel('Curvature |K|')
    axs[2].set_xlabel('x')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    output_path = r"d:\Research project\benchmark_results.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nSaved benchmark plot to {output_path}")


if __name__ == "__main__":
    run_chirp_benchmark()
    run_colored_noise_lorenz_benchmark()
    generate_benchmark_plot()


if __name__ == "__main__":
    run_chirp_benchmark()
    run_colored_noise_lorenz_benchmark()
    generate_benchmark_plot()
