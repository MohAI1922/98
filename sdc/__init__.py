"""
Smooth Discrete Calculus (SDC) & Integral Derivative Manifolds (IDM)
===================================================================
A scale-adaptive integral kernel operator and differential manifold
for noise-robust numerical differentiation.
"""

from .operator import SmoothDiscreteCalculus
from .graph import GraphIDM
from .kernels import compute_kernel, compute_boundary_kernel, get_normalization_constant
from .noise import (
    generate_white_noise,
    generate_ornstein_uhlenbeck_noise,
    theoretical_variance_ou
)
from .utils import estimate_noise_std, estimate_correlation_length

__all__ = [
    "SmoothDiscreteCalculus",
    "GraphIDM",
    "compute_kernel",
    "compute_boundary_kernel",
    "get_normalization_constant",
    "generate_white_noise",
    "generate_ornstein_uhlenbeck_noise",
    "theoretical_variance_ou",
    "estimate_noise_std",
    "estimate_correlation_length",
]

__version__ = "0.2.0"
