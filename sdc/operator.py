"""
SmoothDiscreteCalculus: The Integral Derivative Manifold (IDM) Engine.
Implements:
- Parametric IDM operators (beta >= 0)
- Commutative spatial derivatives d/dx D_eps (Theorem 1)
- Scale derivatives d/deps D_eps and Ridgeline tracking (Theorem 2)
- Gaussian curvature K(x, eps) for regularity detection (Theorem 3)
- Colored noise variance adaptation (Theorem 4)
- Boundary-adapted edge kernels
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from .kernels import compute_kernel, compute_boundary_kernel, get_normalization_constant
from .utils import estimate_noise_std


class SmoothDiscreteCalculus:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        noise_std: float | None = None
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.x) != len(self.y):
            raise ValueError("x and y arrays must have identical length.")
        if len(self.x) < 4:
            raise ValueError("Domain must contain at least 4 points.")
        self.dx = float(self.x[1] - self.x[0])
        self.sigma = float(noise_std if noise_std is not None else estimate_noise_std(self.y))

    def compute(
        self,
        epsilon: float,
        beta: float = 0.0,
        use_boundary_correction: bool = True
    ) -> np.ndarray:
        """
        Computes the IDM derivative D_eps^(beta) f(x).
        Uses FFT convolution for interior points and exact boundary kernels near edges.
        """
        k_len = max(1, int(np.ceil(epsilon / self.dx)))
        t = np.arange(-k_len, k_len + 1) * self.dx
        kernel = compute_kernel(t, epsilon, beta=beta, dx=self.dx) * self.dx

        # Interior convolution
        result = fftconvolve(self.y, kernel[::-1], mode='same')

        if use_boundary_correction and k_len < len(self.x) // 2:
            n = len(self.x)
            # Left boundary
            for i in range(k_len):
                c_left = i * self.dx / epsilon
                c_right = min(1.0, (n - 1 - i) * self.dx / epsilon)
                k_bound = compute_boundary_kernel(t, epsilon, c_left, c_right) * self.dx
                idx_start = max(0, i - k_len)
                idx_end = min(n, i + k_len + 1)
                t_sub = self.x[idx_start:idx_end] - self.x[i]
                k_sub = compute_boundary_kernel(t_sub, epsilon, c_left, c_right) * self.dx
                result[i] = np.sum(self.y[idx_start:idx_end] * k_sub)

            # Right boundary
            for i in range(n - k_len, n):
                c_left = min(1.0, i * self.dx / epsilon)
                c_right = (n - 1 - i) * self.dx / epsilon
                idx_start = max(0, i - k_len)
                idx_end = min(n, i + k_len + 1)
                t_sub = self.x[idx_start:idx_end] - self.x[i]
                k_sub = compute_boundary_kernel(t_sub, epsilon, c_left, c_right) * self.dx
                result[i] = np.sum(self.y[idx_start:idx_end] * k_sub)

        return result

    def compute_spatial_derivative(
        self,
        epsilon: float,
        beta: float = 0.0
    ) -> np.ndarray:
        """
        Computes partial_x D_eps f(x) using the Theorem 1 recursion:
            partial_x D_eps f(x) = D_eps [f'(x)]
        """
        d_val = self.compute(epsilon, beta=beta)
        # Numerical gradient of the smooth manifold
        return np.gradient(d_val, self.dx)

    def compute_scale_derivative(
        self,
        epsilon: float
    ) -> np.ndarray:
        """
        Computes analytical scale derivative partial_eps D_eps f(x) for beta=0 (Theorem 2):
            partial_eps D_eps f(x) = (3 / (2*eps^2)) * [f(x+eps) - f(x-eps)] - (3 / eps) * D_eps f(x)
        """
        k_len = max(1, int(np.round(epsilon / self.dx)))
        d_val = self.compute(epsilon, beta=0.0, use_boundary_correction=False)
        f_plus = np.roll(self.y, -k_len)
        f_minus = np.roll(self.y, k_len)

        scale_deriv = (3.0 / (2.0 * (epsilon**2))) * (f_plus - f_minus) - (3.0 / epsilon) * d_val

        # Mask boundary wraps
        scale_deriv[:k_len] = scale_deriv[k_len]
        scale_deriv[-k_len:] = scale_deriv[-k_len - 1]
        return scale_deriv

    def compute_manifold(
        self,
        eps_grid: np.ndarray,
        beta: float = 0.0
    ) -> np.ndarray:
        """
        Evaluates the 2D surface z = D_eps f(x) over a grid of epsilon scales.
        Returns array of shape (len(eps_grid), len(self.x)).
        """
        manifold = np.zeros((len(eps_grid), len(self.x)), dtype=np.float64)
        for i, eps in enumerate(eps_grid):
            manifold[i, :] = self.compute(eps, beta=beta)
        return manifold

    def compute_gaussian_curvature(
        self,
        eps_grid: np.ndarray,
        beta: float = 0.0
    ) -> np.ndarray:
        """
        Computes the Gaussian Curvature K(x, eps) across the IDM surface (Theorem 3):
            K = (D_xx * D_epseps - D_xeps^2) / (1 + D_x^2 + D_eps^2)^2
        """
        z = self.compute_manifold(eps_grid, beta=beta)
        # z has shape (num_eps, num_x)
        deps = float(eps_grid[1] - eps_grid[0]) if len(eps_grid) > 1 else 1.0

        # Compute partial derivatives via central differences on the smooth surface
        d_eps, d_x = np.gradient(z, deps, self.dx)
        d_xx = np.gradient(d_x, self.dx, axis=1)
        d_epseps = np.gradient(d_eps, deps, axis=0)
        d_xeps = np.gradient(d_x, deps, axis=0)

        numerator = d_xx * d_epseps - (d_xeps ** 2)
        denominator = (1.0 + d_x**2 + d_eps**2) ** 2
        return numerator / np.maximum(denominator, 1e-12)

    def compute_ridgeline(
        self,
        eps_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the zero scale-sensitivity ridgeline Gamma: {(x, eps*(x)) : partial_eps D_eps f(x) = 0}.
        Returns (eps_ridgeline, derivative_at_ridgeline).
        """
        z = self.compute_manifold(eps_grid, beta=0.0)
        deps = float(eps_grid[1] - eps_grid[0]) if len(eps_grid) > 1 else 1.0
        d_eps = np.gradient(z, deps, axis=0)

        # For each spatial location x, find the epsilon where |d_eps| is minimized
        min_indices = np.argmin(np.abs(d_eps), axis=0)
        optimal_eps = eps_grid[min_indices]
        optimal_deriv = np.array([z[min_indices[j], j] for j in range(len(self.x))])

        return optimal_eps, optimal_deriv

    def compute_adaptive(
        self,
        initial_eps: float | None = None,
        beta: float = 0.0,
        noise_type: str = "white",
        tau: float | None = None
    ) -> np.ndarray:
        """
        Self-tuning optimal scale differentiator (Theorems 2, 4, 5, 6).
        Selects eps*(x) dynamically balancing local curvature (f''') and noise variance.
        Uses a two-pass pilot estimator to obtain an uncorrupted estimate of f'''(x).
        """
        span = self.x[-1] - self.x[0]
        if initial_eps is None:
            initial_eps = max(4.0 * self.dx, span * 0.05)

        # Pass 1: Robust Pilot estimate of higher derivatives (Theorem 6)
        try:
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(self.x, self.y, s=len(self.x) * (self.sigma ** 2))
            f_triple = spl.derivative(n=3)(self.x)
        except Exception:
            # Fallback to smoothed finite differences if spline fitting fails
            d_pilot = self.compute(initial_eps, beta=beta)
            d_pilot_xx = np.gradient(np.gradient(d_pilot, self.dx), self.dx)
            f_triple = d_pilot_xx

        f_triple_sq = np.maximum(f_triple**2, 1e-4)

        if noise_type == "white" or tau is None or tau <= 0.0:
            # AMISE-optimal scale constant c_opt(beta) = (3 R) / (4 mu4^2)
            # For beta=0: c_opt = 112.5. For beta=1: c_opt = 315.0
            c_opt = 315.0 if beta >= 1.0 else 112.5
            eps_opt = ((c_opt * (self.sigma**2) * self.dx) / f_triple_sq) ** (1.0 / 7.0)
        else:
            # Colored noise formula (Theorem 5, Eq. 534):
            # eps*(x) = ( 15 * sigma^2 * dx / (tau_c * f'''^2) )^(1/5)
            eps_opt = ((15.0 * (self.sigma**2) * self.dx) / (tau * f_triple_sq)) ** (1.0 / 5.0)

        # Smooth epsilon trajectory to prevent high-frequency scale chatter
        kernel_smooth = np.ones(11) / 11.0
        eps_opt = np.convolve(eps_opt, kernel_smooth, mode='same')
        eps_opt = np.clip(eps_opt, 4.0 * self.dx, min(0.35, span * 0.15))

        # Pass 2: Pointwise evaluation at optimal scales eps*(x)
        result = np.zeros_like(self.y)
        for i, eps in enumerate(eps_opt):
            k_len = max(1, int(np.ceil(eps / self.dx)))
            idx_start = max(0, i - k_len)
            idx_end = min(len(self.y), i + k_len + 1)
            t = self.x[idx_start:idx_end] - self.x[i]
            k_sub = compute_kernel(t, eps, beta=beta, dx=self.dx) * self.dx
            result[i] = np.sum(self.y[idx_start:idx_end] * k_sub)

        return result
