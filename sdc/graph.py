"""
Graph Integral Derivative Manifolds (G-IDM)
===========================================
Coordinate-free and noise-robust gradient calculus on:
- Non-uniform 2D/3D point clouds
- Irregular sensor networks
- Weighted graphs with shortest-path geodesic distances
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import shortest_path


class GraphIDM:
    def __init__(
        self,
        coords: np.ndarray,
        signal: np.ndarray,
        dist_matrix: np.ndarray | None = None,
        noise_std: float | None = None,
        intrinsic_dim: int | None = None
    ):
        """
        Parameters:
        -----------
        coords: (N, D) array of node coordinates (ambient spatial or spectral embedding)
        signal: (N,) array of scalar measurements on nodes
        dist_matrix: Optional (N, N) geodesic/graph shortest-path distance matrix.
                     If None, Euclidean distances in coords space are computed.
        noise_std: Optional noise standard deviation.
        intrinsic_dim: Intrinsic manifold dimension d <= D. If None, defaults to ambient D.
                       When d < D, gradient is computed on the local tangent hyperplane
                       spanned by the top d eigenvectors of the Metric Moment Tensor M_eps.
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        if self.coords.ndim == 1:
            self.coords = self.coords[:, np.newaxis]
        self.signal = np.asarray(signal, dtype=np.float64)
        self.n_nodes, self.dim = self.coords.shape
        self.intrinsic_dim = int(intrinsic_dim) if intrinsic_dim is not None else self.dim

        if len(self.signal) != self.n_nodes:
            raise ValueError(f"Signal length {len(self.signal)} must match number of nodes {self.n_nodes}.")

        if dist_matrix is not None:
            self.dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
        else:
            self.dist_matrix = cdist(self.coords, self.coords)

        self.sigma = noise_std if noise_std is not None else self._estimate_noise()

    @classmethod
    def from_point_cloud_geodesic(
        cls,
        coords: np.ndarray,
        signal: np.ndarray,
        k_neighbors: int = 10,
        noise_std: float | None = None,
        intrinsic_dim: int | None = None
    ) -> GraphIDM:
        """Constructs GraphIDM using graph shortest-path geodesic distance matrix."""
        n = len(coords)
        tree = cKDTree(coords)
        dists, indices = tree.query(coords, k=k_neighbors + 1)
        adj = dok_matrix((n, n), dtype=float)
        for i in range(n):
            for k in range(1, k_neighbors + 1):
                j = indices[i, k]
                d = dists[i, k]
                adj[i, j] = d
                adj[j, i] = d
        geo_dist = shortest_path(adj.tocsr(), directed=False)
        return cls(coords, signal, dist_matrix=geo_dist, noise_std=noise_std, intrinsic_dim=intrinsic_dim)

    @classmethod
    def from_adjacency_matrix(
        cls,
        adj_matrix: np.ndarray,
        signal: np.ndarray,
        embedding_dim: int = 3,
        noise_std: float | None = None
    ) -> GraphIDM:
        """
        Constructs GraphIDM for non-geometric / abstract graphs using
        Normalized Laplacian Spectral Embedding coordinates.
        """
        from scipy.sparse.csgraph import laplacian
        from scipy.linalg import eigh
        n = adj_matrix.shape[0]
        # Normalized symmetric Laplacian
        L_sym = laplacian(adj_matrix, normed=True)
        # Smallest non-trivial eigenvectors
        w, v = eigh(L_sym)
        # Exclude constant trivial eigenvector at index 0
        coords = v[:, 1:embedding_dim + 1]
        geo_dist = shortest_path(adj_matrix, directed=False)
        return cls(coords, signal, dist_matrix=geo_dist, noise_std=noise_std, intrinsic_dim=embedding_dim)

    def _estimate_noise(self) -> float:
        """Estimates noise standard deviation from nearest-neighbor residuals."""
        dists = np.copy(self.dist_matrix)
        np.fill_diagonal(dists, np.inf)
        nn_indices = np.argmin(dists, axis=1)
        diffs = self.signal - self.signal[nn_indices]
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        return float(1.4826 * mad / np.sqrt(2.0))

    def compute_gradient(
        self,
        epsilon: float,
        beta: float = 1.0,
        reg_lambda: float = 1e-6
    ) -> np.ndarray:
        """
        Computes spatial gradient vector nabla_eps^G f(v_i) in R^D for each node.
        If intrinsic_dim < D, projects onto local tangent plane.
        """
        gradients = np.zeros((self.n_nodes, self.dim), dtype=np.float64)
        d_sub = self.intrinsic_dim

        for i in range(self.n_nodes):
            dists_i = self.dist_matrix[i]
            mask = (dists_i > 1e-12) & (dists_i <= epsilon)

            if np.sum(mask) < d_sub:
                k_fallback = min(self.n_nodes - 1, d_sub + 2)
                sort_idx = np.argsort(dists_i)
                mask = np.zeros(self.n_nodes, dtype=bool)
                mask[sort_idx[1:k_fallback + 1]] = True
                eps_local = dists_i[sort_idx[k_fallback]] * 1.05
            else:
                eps_local = epsilon

            dists_sub = dists_i[mask]
            u = dists_sub / eps_local
            weights = (1.0 - np.clip(u**2, 0.0, 1.0)) ** beta

            v_ij = self.coords[mask] - self.coords[i]  # shape: (K, D)
            df = self.signal[mask] - self.signal[i]   # shape: (K,)

            # Moment tensor M_eps = sum_j w_j * v_j v_j^T (D x D)
            M = (v_ij.T * weights) @ v_ij
            b = np.sum((weights * df)[:, np.newaxis] * v_ij, axis=0)

            if d_sub < self.dim:
                # Tangent-space projection via eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(M)
                # Top d eigenvectors span local tangent space
                V_tangent = eigvecs[:, -d_sub:]
                M_tangent = V_tangent.T @ M @ V_tangent
                b_tangent = V_tangent.T @ b
                reg = reg_lambda * (np.trace(M_tangent) / max(1, d_sub) + 1e-10)
                M_reg = M_tangent + reg * np.eye(d_sub)
                grad_tangent = np.linalg.solve(M_reg, b_tangent)
                gradients[i] = V_tangent @ grad_tangent
            else:
                # Full ambient dimension
                reg = reg_lambda * (np.trace(M) / max(1, self.dim) + 1e-10)
                M_reg = M + reg * np.eye(self.dim)
                gradients[i] = np.linalg.solve(M_reg, b)

        return gradients

    def compute_gradient_magnitude(
        self,
        epsilon: float,
        beta: float = 1.0
    ) -> np.ndarray:
        """Computes Euclidean magnitude ||nabla_eps^G f|| for all nodes."""
        grads = self.compute_gradient(epsilon=epsilon, beta=beta)
        return np.linalg.norm(grads, axis=1)

    def compute_directional_derivative(
        self,
        target_indices: np.ndarray,
        epsilon: float,
        beta: float = 1.0
    ) -> np.ndarray:
        """Computes directional derivative along directed edges (i, target_indices[i])."""
        grads = self.compute_gradient(epsilon=epsilon, beta=beta)
        dir_derivs = np.zeros(self.n_nodes, dtype=np.float64)

        for i, j in enumerate(target_indices):
            disp = self.coords[j] - self.coords[i]
            dist = np.linalg.norm(disp)
            if dist > 1e-12:
                unit_v = disp / dist
                dir_derivs[i] = float(np.dot(grads[i], unit_v))

        return dir_derivs
