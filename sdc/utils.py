"""
Utility methods and noise estimation for IDM.
"""

from __future__ import annotations
import numpy as np


def estimate_noise_std(y: np.ndarray) -> float:
    """
    Robust noise standard deviation estimation using Median Absolute Deviation (MAD)
    of the second-order differences:
        delta2 = y[i+2] - 2*y[i+1] + y[i]
    For white noise, Var(delta2) = (1 + 4 + 1) * sigma^2 = 6 * sigma^2.
    """
    if len(y) < 3:
        return 0.0
    diff2 = np.diff(y, n=2)
    # MAD estimator: sigma = 1.4826 * MAD / sqrt(6)
    med = np.median(diff2)
    mad = np.median(np.abs(diff2 - med))
    sigma_est = (1.4826 * mad) / np.sqrt(6.0)
    return float(sigma_est)


def estimate_correlation_length(y: np.ndarray, dx: float, max_lags: int = 50) -> float:
    """
    Estimates the empirical noise correlation length tau via normalized autocorrelation
    of high-pass filtered residuals.
    """
    diff2 = np.diff(y, n=2)
    diff2 = diff2 - np.mean(diff2)
    var = np.var(diff2)
    if var < 1e-15:
        return 1e-6
    lags = min(max_lags, len(diff2) // 2)
    autocorr = [np.corrcoef(diff2[:-k], diff2[k:])[0, 1] for k in range(1, lags)]
    # Find 1/e decay crossing
    target = 1.0 / np.e
    for idx, r in enumerate(autocorr):
        if r <= target:
            return float((idx + 1) * dx)
    return float(lags * dx)
