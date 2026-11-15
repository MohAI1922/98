"""
Graph IDM Benchmarking Suite
Evaluates G-IDM against established baselines:
  1. Naive k-NN Graph Finite Differences
  2. Graph Tikhonov Regularized Differences
  3. Moving Least Squares (MLS) Local Linear Regression
  4. Geodesic Graph IDM (Proposed)
Across:
  - Benchmark 1: Irregular 2D Sensor Network (N = 800)
  - Benchmark 2: 3D Spherical Surface Point Cloud (N = 1000)
  - Benchmark 3: Folded Swiss Roll Manifold (N = 1200, Geodesic vs Chordal Metric)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix, eye
from scipy.sparse.linalg import spsolve
from sdc import GraphIDM


def naive_knn_gradient(coords: np.ndarray, signal: np.ndarray, k: int = 8) -> np.ndarray:
    """Standard k-NN graph directional finite difference baseline."""
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)
    grads = np.zeros_like(coords)

    for i in range(len(coords)):
        disp = coords[indices[i, 1:]] - coords[i]
        d_vals = dists[i, 1:]
        df = signal[indices[i, 1:]] - signal[i]
        unit_disp = disp / np.maximum(d_vals[:, np.newaxis], 1e-12)
        grads[i] = np.mean((df / np.maximum(d_vals, 1e-12))[:, np.newaxis] * unit_disp, axis=0)

    return grads


def compute_mls_gradient(coords: np.ndarray, signal: np.ndarray, h: float = 0.15) -> np.ndarray:
    """Moving Least Squares (MLS) local linear polynomial regression."""
    n, d = coords.shape
    grads = np.zeros_like(coords)
    for i in range(n):
        disp = coords - coords[i]
        dists = np.linalg.norm(disp, axis=1)
        mask = (dists > 1e-12) & (dists <= 2.5 * h)
        if np.sum(mask) < d:
            mask = np.argsort(dists)[1:d + 2]
        w = np.exp(-(dists[mask] / h)**2)
        v = disp[mask]
        df = signal[mask] - signal[i]
        M = (v.T * w) @ v + 1e-5 * np.eye(d)
        b = np.sum((w * df)[:, np.newaxis] * v, axis=0)
        grads[i] = np.linalg.solve(M, b)
    return grads


def compute_graph_tikhonov_gradient(coords: np.ndarray, signal: np.ndarray, k: int = 8, lam: float = 0.5) -> np.ndarray:
    """Graph Tikhonov smoothing followed by local difference."""
    n = len(coords)
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)
    adj = dok_matrix((n, n), dtype=float)
    for i in range(n):
        for idx in range(1, k + 1):
            j = indices[i, idx]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    W = adj.tocsr()
    deg = np.array(W.sum(axis=1)).flatten()
    L = dok_matrix((n, n), dtype=float)
    for i in range(n):
        L[i, i] = deg[i]
    L = L.tocsr() - W
    A = eye(n) + lam * L
    y_smooth = spsolve(A.tocsc(), signal)
    return naive_knn_gradient(coords, y_smooth, k=k)


def run_sensor_network_benchmark():
    print("\n" + "=" * 95)
    print("GRAPH BENCHMARK 1: Irregular 2D Sensor Network (N = 800 Nodes)")
    print("=" * 95)
    rng = np.random.default_rng(2024)
    n = 800
    coords = rng.uniform(0.05, 0.95, size=(n, 2))

    x = coords[:, 0]
    y = coords[:, 1]
    f_clean = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    grad_x_true = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    grad_y_true = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    true_grads = np.column_stack([grad_x_true, grad_y_true])

    noise_levels = [0.05, 0.15, 0.30]
    interior = (x > 0.15) & (x < 0.85) & (y > 0.15) & (y < 0.85)

    header = f"{'Noise (sigma)':<15} | {'k-NN FD (k=8)':<16} | {'Graph Tikhonov':<17} | {'Moving Least Sq':<18} | {'G-IDM (Proposed)':<18}"
    print(header)
    print("-" * len(header))

    for sigma in noise_levels:
        noisy_signal = f_clean + rng.normal(0.0, sigma, size=n)

        # 1. k-NN FD
        grad_knn = naive_knn_gradient(coords, noisy_signal, k=8)
        err_knn = float(np.sqrt(np.mean(np.linalg.norm(grad_knn[interior] - true_grads[interior], axis=1)**2)))

        # 2. Graph Tikhonov
        grad_tik = compute_graph_tikhonov_gradient(coords, noisy_signal, k=8, lam=0.5)
        err_tik = float(np.sqrt(np.mean(np.linalg.norm(grad_tik[interior] - true_grads[interior], axis=1)**2)))

        # 3. MLS
        grad_mls = compute_mls_gradient(coords, noisy_signal, h=0.15)
        err_mls = float(np.sqrt(np.mean(np.linalg.norm(grad_mls[interior] - true_grads[interior], axis=1)**2)))

        # 4. G-IDM
        gidm = GraphIDM(coords, noisy_signal, noise_std=sigma)
        grad_gidm = gidm.compute_gradient(epsilon=0.18, beta=1.0)
        err_gidm = float(np.sqrt(np.mean(np.linalg.norm(grad_gidm[interior] - true_grads[interior], axis=1)**2)))

        print(f"AWGN sigma={sigma:<6.2f} | {err_knn:<16.4f} | {err_tik:<17.4f} | {err_mls:<18.4f} | {err_gidm:<18.4f}")


def run_sphere_benchmark():
    print("\n" + "=" * 95)
    print("GRAPH BENCHMARK 2: 3D Spherical Surface Point Cloud (N = 1000 Nodes)")
    print("=" * 95)
    rng = np.random.default_rng(42)
    n = 1000
    vecs = rng.normal(size=(n, 3))
    coords = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Signal f(x, y, z) = x * y
    f_clean = x * y
    ambient_grad = np.column_stack([y, x, np.zeros_like(x)])
    tangent_proj = np.sum(ambient_grad * coords, axis=1, keepdims=True)
    true_tangent_grad = ambient_grad - tangent_proj * coords

    noise_levels = [0.05, 0.15, 0.30]
    header = f"{'Noise (sigma)':<15} | {'k-NN FD':<16} | {'Moving Least Sq':<18} | {'G-IDM Ambient':<18} | {'G-IDM Tangent Proj':<20}"
    print(header)
    print("-" * len(header))

    for sigma in noise_levels:
        noisy_signal = f_clean + rng.normal(0.0, sigma, size=n)

        # 1. k-NN
        grad_knn = naive_knn_gradient(coords, noisy_signal, k=10)
        grad_knn_tan = grad_knn - np.sum(grad_knn * coords, axis=1, keepdims=True) * coords
        err_knn = float(np.sqrt(np.mean(np.linalg.norm(grad_knn_tan - true_tangent_grad, axis=1)**2)))

        # 2. MLS
        grad_mls = compute_mls_gradient(coords, noisy_signal, h=0.30)
        grad_mls_tan = grad_mls - np.sum(grad_mls * coords, axis=1, keepdims=True) * coords
        err_mls = float(np.sqrt(np.mean(np.linalg.norm(grad_mls_tan - true_tangent_grad, axis=1)**2)))

        # 3. G-IDM Ambient
        gidm_amb = GraphIDM(coords, noisy_signal, noise_std=sigma)
        grad_amb = gidm_amb.compute_gradient(epsilon=0.35, beta=1.0)
        grad_amb_tan = grad_amb - np.sum(grad_amb * coords, axis=1, keepdims=True) * coords
        err_amb = float(np.sqrt(np.mean(np.linalg.norm(grad_amb_tan - true_tangent_grad, axis=1)**2)))

        # 4. G-IDM Tangent Proj (intrinsic_dim = 2)
        gidm_tan = GraphIDM(coords, noisy_signal, noise_std=sigma, intrinsic_dim=2)
        grad_tan = gidm_tan.compute_gradient(epsilon=0.35, beta=1.0)
        err_tan = float(np.sqrt(np.mean(np.linalg.norm(grad_tan - true_tangent_grad, axis=1)**2)))

        print(f"AWGN sigma={sigma:<6.2f} | {err_knn:<16.4f} | {err_mls:<18.4f} | {err_amb:<18.4f} | {err_tan:<20.4f}")


def run_swiss_roll_benchmark():
    print("\n" + "=" * 95)
    print("GRAPH BENCHMARK 3: Folded Swiss Roll Manifold (Geodesic vs Ambient Euclidean Metric)")
    print("=" * 95)
    rng = np.random.default_rng(42)
    n = 1200
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0, 1, size=n))
    y = 20.0 * rng.uniform(0, 1, size=n)
    x = t * np.cos(t)
    z = t * np.sin(t)
    coords = np.column_stack([x, y, z])

    f_clean = np.sin(t / 2.0) * np.cos(y / 5.0)
    sigma = 0.08
    noisy_signal = f_clean + rng.normal(0.0, sigma, size=n)

    # 1. Euclidean MLS
    grad_mls_euc = compute_mls_gradient(coords, noisy_signal, h=2.5)

    # 2. Geodesic G-IDM
    gidm_geo = GraphIDM.from_point_cloud_geodesic(coords, noisy_signal, k_neighbors=8, noise_std=sigma, intrinsic_dim=2)
    grad_gidm_geo = gidm_geo.compute_gradient(epsilon=6.0, beta=1.0)

    # True tangent gradient
    dt_norm = np.sqrt(1.0 + t**2)
    e_t = np.column_stack([(np.cos(t) - t*np.sin(t))/dt_norm, np.zeros_like(t), (np.sin(t) + t*np.cos(t))/dt_norm])
    e_y = np.column_stack([np.zeros_like(t), np.ones_like(t), np.zeros_like(t)])
    df_dt_arc = (0.5 * np.cos(t / 2.0) * np.cos(y / 5.0)) / dt_norm
    df_dy_arc = -0.2 * np.sin(t / 2.0) * np.sin(y / 5.0)
    true_tangent = df_dt_arc[:, np.newaxis] * e_t + df_dy_arc[:, np.newaxis] * e_y

    interior = (t > 2.0 * np.pi) & (t < 4.0 * np.pi)
    err_mls = float(np.sqrt(np.mean(np.linalg.norm(grad_mls_euc[interior] - true_tangent[interior], axis=1)**2)))
    err_gidm = float(np.sqrt(np.mean(np.linalg.norm(grad_gidm_geo[interior] - true_tangent[interior], axis=1)**2)))

    print(f"Chordal Euclidean MLS Error:  {err_mls:.4f}")
    print(f"Geodesic G-IDM Error:         {err_gidm:.4f}  (Improvement: {err_mls/err_gidm:.1f}x reduction in RMSE)")


def generate_graph_benchmark_plot():
    """Generates 2D vector field visualization comparing G-IDM with k-NN FD."""
    rng = np.random.default_rng(42)
    n = 600
    coords = rng.uniform(0.1, 0.9, size=(n, 2))
    x, y = coords[:, 0], coords[:, 1]
    f_clean = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    grad_x_true = 2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    grad_y_true = -2.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)

    sigma = 0.12
    noisy_signal = f_clean + rng.normal(0.0, sigma, size=n)

    gidm = GraphIDM(coords, noisy_signal, noise_std=sigma)
    grad_gidm = gidm.compute_gradient(epsilon=0.18, beta=1.0)
    grad_knn = naive_knn_gradient(coords, noisy_signal, k=8)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(x, y, c=f_clean, cmap='viridis', s=25, alpha=0.8)
    axs[0].quiver(x[::4], y[::4], grad_x_true[::4], grad_y_true[::4], color='red', scale=35)
    axs[0].set_title("Ground Truth Potential & Gradient")

    axs[1].scatter(x, y, c=noisy_signal, cmap='viridis', s=25, alpha=0.8)
    axs[1].quiver(x[::4], y[::4], grad_knn[::4, 0], grad_knn[::4, 1], color='darkorange', scale=35)
    axs[1].set_title(f"Naive k-NN Difference (sigma={sigma})")

    axs[2].scatter(x, y, c=noisy_signal, cmap='viridis', s=25, alpha=0.8)
    axs[2].quiver(x[::4], y[::4], grad_gidm[::4, 0], grad_gidm[::4, 1], color='blue', scale=35)
    axs[2].set_title("G-IDM Reconstruction (Proposed)")

    for ax in axs:
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.05, 0.95)
        ax.set_aspect('equal')

    plt.tight_layout()
    output_path = r"d:\Research project\graph_benchmark_results.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nSaved graph benchmark plot to {output_path}")


if __name__ == "__main__":
    run_sensor_network_benchmark()
    run_sphere_benchmark()
    run_swiss_roll_benchmark()
    generate_graph_benchmark_plot()
