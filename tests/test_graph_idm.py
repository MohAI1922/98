"""
Test suite for Graph Integral Derivative Manifolds (G-IDM).
Verifies:
1. Theorem G1: Exact linear gradient reproduction on arbitrary non-uniform point clouds.
2. Asymptotic accuracy on smooth 2D quadratic fields.
3. Noise variance suppression on 3D point clouds.
"""

import numpy as np
from sdc import GraphIDM


def test_theorem_g1_exact_linear_recovery():
    """
    Verifies Theorem G1:
    For an affine field f(x, y) = a*x + b*y + c, G-IDM recovers
    nabla f == (a, b) exactly on random, irregular point clouds.
    """
    rng = np.random.default_rng(42)
    n = 300
    # Random ungridded coordinates
    coords = rng.uniform(0.0, 2.0, size=(n, 2))
    a_true = 3.5
    b_true = -2.1
    c_const = 1.2
    signal = a_true * coords[:, 0] + b_true * coords[:, 1] + c_const

    gidm = GraphIDM(coords, signal, noise_std=0.0)
    eps = 0.35
    grads = gidm.compute_gradient(epsilon=eps, beta=1.0)

    # Check on interior nodes away from edge bounding box
    interior = (coords[:, 0] > 0.4) & (coords[:, 0] < 1.6) & (coords[:, 1] > 0.4) & (coords[:, 1] < 1.6)
    interior_grads = grads[interior]

    err_x = np.max(np.abs(interior_grads[:, 0] - a_true))
    err_y = np.max(np.abs(interior_grads[:, 1] - b_true))

    print(f"Theorem G1 Linear Recovery: err_x = {err_x:.6e}, err_y = {err_y:.6e}")
    assert err_x < 1e-4, f"Theorem G1 failed on x-component: err = {err_x}"
    assert err_y < 1e-4, f"Theorem G1 failed on y-component: err = {err_y}"


def test_quadratic_gradient_accuracy():
    """
    Verifies accuracy on 2D quadratic field f(x, y) = x^2 + 2*y^2.
    True gradient is (2x, 4y).
    """
    rng = np.random.default_rng(101)
    n = 600
    coords = rng.uniform(-1.0, 1.0, size=(n, 2))
    signal = coords[:, 0]**2 + 2.0 * coords[:, 1]**2
    true_grads = np.column_stack([2.0 * coords[:, 0], 4.0 * coords[:, 1]])

    gidm = GraphIDM(coords, signal, noise_std=0.0)
    eps = 0.20
    est_grads = gidm.compute_gradient(epsilon=eps, beta=1.0)

    interior = (np.abs(coords[:, 0]) < 0.6) & (np.abs(coords[:, 1]) < 0.6)
    diff = np.linalg.norm(est_grads[interior] - true_grads[interior], axis=1)
    mean_err = np.mean(diff)

    print(f"Quadratic Field Interior Mean Gradient Error: {mean_err:.4f}")
    assert mean_err < 0.12, f"Quadratic gradient error too high: {mean_err}"


def test_noise_robustness_on_3d_point_cloud():
    """
    Verifies noise suppression on a 3D spherical point cloud with Gaussian noise.
    """
    rng = np.random.default_rng(77)
    n = 500
    # Random points on sphere
    vecs = rng.normal(size=(n, 3))
    coords = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # Signal f(x, y, z) = z
    f_clean = coords[:, 2]
    sigma = 0.15
    noisy_signal = f_clean + rng.normal(0.0, sigma, size=n)

    gidm = GraphIDM(coords, noisy_signal, noise_std=sigma)
    est_grads = gidm.compute_gradient(epsilon=0.45, beta=1.0)

    # True gradient of z in ambient space is [0, 0, 1]
    true_grads = np.zeros((n, 3))
    true_grads[:, 2] = 1.0

    # Project to tangential plane: for unit sphere, ambient gradient of z restricted to sphere
    # grad_S(z) = [0, 0, 1] - z * [x, y, z]
    tangent_true = true_grads - coords[:, 2:3] * coords

    # Project estimated gradient to tangent space
    est_tangent = est_grads - np.sum(est_grads * coords, axis=1, keepdims=True) * coords

    err = np.mean(np.linalg.norm(est_tangent - tangent_true, axis=1))
    print(f"3D Spherical Surface Gradient Error under Noise: {err:.4f}")
    assert err < 0.50, f"Sphere gradient error under noise too high: {err}"


if __name__ == "__main__":
    test_theorem_g1_exact_linear_recovery()
    print("PASS: Theorem G1 (Exact Linear Recovery on Random Point Clouds)")
    test_quadratic_gradient_accuracy()
    print("PASS: Quadratic 2D Field Gradient Accuracy")
    test_noise_robustness_on_3d_point_cloud()
    print("PASS: 3D Point Cloud Noise Robustness")
    print("\nALL GRAPH IDM TESTS PASSED.")
