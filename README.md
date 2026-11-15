# Smooth Discrete Calculus & Integral Derivative Manifolds (IDM)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)]()
[![Tests: 8 passed](https://img.shields.io/badge/Tests-8%20Passed-success.svg)]()

**Author:** Muhammed M. El-Sayed  
*Department of Electrical and Computer Engineering / Applied Mathematics*

This repository contains the complete open-source scientific implementation, benchmark suites, and publication manuscripts for the two-part research agenda on **Integral Derivative Manifolds (IDM & G-IDM)**.

---

## 📚 Publications

### Paper 1 (Continuous Euclidean IDM)
* **Title:** *Differential Geometry of Integral Derivative Manifolds: Noise Decoupling, Colored Noise Estimation, and Optimal Scale Calibration in Euclidean Domains*
* **Target Venues:** *SIAM Journal on Numerical Analysis (SINUM)* / *IEEE Transactions on Signal Processing (TSP)*
* **Manuscript:** [`paper1_manuscript.pdf`](paper1_manuscript.pdf) (22 pages)
* **LaTeX Source:** [`paper1_manuscript.tex`](paper1_manuscript.tex)
* **BibTeX:** [`paper1_references.bib`](paper1_references.bib)
* **Submission Cover Letter:** [`submissions/COVER_LETTER_PAPER1.pdf`](submissions/COVER_LETTER_PAPER1.pdf)

### Paper 2 (Discrete Graph & Submanifold G-IDM)
* **Title:** *Graph Integral Derivative Manifolds: Metric Moment Tensors, Non-Euclidean Scale-Space Geometry, and Noise-Proof Field Calculus on Irregular Networks and Point Clouds*
* **Target Venues:** *IEEE Transactions on Signal and Information Processing over Networks (TSIPN)* / *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* / *SIAM Journal on Mathematics of Data Science (SIMODS)*
* **Manuscript:** [`paper2_manuscript.pdf`](paper2_manuscript.pdf) (15 pages)
* **LaTeX Source:** [`paper2_manuscript.tex`](paper2_manuscript.tex)
* **BibTeX:** [`paper2_references.bib`](paper2_references.bib)
* **Submission Cover Letter:** [`submissions/COVER_LETTER_PAPER2.pdf`](submissions/COVER_LETTER_PAPER2.pdf)

---

## 🚀 Key Theoretical Contributions

```
[Irregular Point Cloud]
        v_j (noisy)
       /  \
      /    \  Naive FD: Var ~ O(sigma^2 / d_ij^2)  --> EXPLODES as d_ij -> 0!
     /  e   \
    v_i ---- v_k
         
[Folded Manifold (Swiss Roll)]
    Fold A ===== v_i =====
              |
              | Chordal Euclidean shortcut: ||x_i - x_j|| is small
              v (Cross-fold bridge causes O(1) gradient distortion!)
    Fold B ===== v_j =====
              |
              +--> Geodesic shortest-path routes along surface: d_G(v_i, v_j) >> ||x_i - x_j||
```

1. **Exact Linear Recovery (Theorem 1 in both papers):** Replaces heuristic differences with an exact Metric Moment Tensor $\mathbf{M}_\epsilon(v_i)$, reproducing linear fields with exact mathematical precision on arbitrary non-uniform point distributions.
2. **Density Inversion Property:** While naive finite differences diverge as node density $N \to \infty$ ($\text{Var} \asymp N^{2/d} \to \infty$), G-IDM variance decays strictly to zero as $\mathcal{O}(\sigma^2 / (k \epsilon^2)) \asymp \mathcal{O}(1/N)$, transforming node density into a statistical asset.
3. **Topological Consistency:** Eliminates the $\mathcal{O}(1)$ cross-fold bias of Euclidean Moving Least Squares (MLS) on thin folded geometries via geodesic Dijkstra routing.
4. **Colored Noise Decoupling:** Proves that temporal noise correlation (Ornstein-Uhlenbeck) softens the derivative singularity from $\mathcal{O}(\epsilon^{-3})$ to $\mathcal{O}(\epsilon^{-1})$.

---

## 🛠️ Installation

```bash
git clone https://github.com/author/sdc-idm.git
cd sdc-idm
pip install -e .
```

### Dependencies
* Python $\ge$ 3.9
* `numpy >= 1.24.0`
* `scipy >= 1.10.0`
* `matplotlib >= 3.7.0`
* `scikit-learn >= 1.2.0`
* `pytest >= 7.0.0`

---

## 🧪 Master Reproduction (One-Click)

To reproduce the entire scientific workflow—including all formal theorem unit tests, continuous Euclidean benchmarks, discrete graph manifold benchmarks, and LaTeX PDF compilations:

```bash
python reproduce_all.py
```

### Running Components Individually

1. **Formal Theorem Unit Tests:**
   ```bash
   python -m pytest tests/ -v
   ```
2. **Paper 1 Continuous Benchmarks:**
   ```bash
   python benchmarks/run_benchmarks.py
   # Generates: benchmark_results.png
   ```
3. **Paper 2 Graph Benchmarks:**
   ```bash
   python benchmarks/run_graph_benchmarks.py
   # Generates: graph_benchmark_results.png
   ```
4. **Compile LaTeX Manuscripts:**
   ```bash
   pdflatex -interaction=nonstopmode paper1_manuscript.tex
   pdflatex -interaction=nonstopmode paper2_manuscript.tex
   ```

---

## 📊 Summary of Experimental Results

### Paper 1: Continuous Numerical Differentiation (Doppler Chirp, AWGN $\sigma = 0.25$)
| Method | Root Mean Squared Error (RMSE) | Relative Advantage |
| :--- | :---: | :---: |
| Central Finite Differences | 101.2102 | Catastrophic noise explosion |
| Savitzky-Golay ($p=2, w=31$) | 0.8682 | Sensitive to scale mismatch |
| Lanczos Differentiator ($M=15$) | 0.5781 | Edge truncation boundary bias |
| Tikhonov Regularization ($L^2$) | 0.5211 | Non-local global coupling |
| **IDM Adaptive (Proposed, $\beta=1$)** | **0.4106** | **246$\times$ vs FD, 2.1$\times$ vs SG** |

### Paper 2: Discrete Graph Differentiation (2D Sensor Network, $N = 800$, $\sigma = 0.30$)
| Method | Root Mean Squared Error (RMSE) | Relative Advantage |
| :--- | :---: | :---: |
| Naive $k$-NN Finite Differences ($k=8$) | 7.8618 | Noise explosion |
| Graph Tikhonov Regularization | 3.2518 | Oversmoothing |
| Moving Least Squares ($h=0.15$) | 1.6332 | Chordal distortion |
| **G-IDM (Proposed, $\epsilon=0.18$)** | **1.0843** | **7.3$\times$ vs FD, 1.5$\times$ vs MLS** |

### Paper 2: Folded Swiss Roll Manifold ($N = 1200$, Noise $\sigma = 0.08$)
| Method | Metric Space Used | Root Mean Squared Error (RMSE) |
| :--- | :--- | :---: |
| Ambient Moving Least Squares (MLS) | Chordal Euclidean $\|\mathbf{x}_j - \mathbf{x}_i\|_{\mathbb{R}^3}$ | 0.1778 |
| **Geodesic G-IDM (Proposed)** | **Geodesic Shortest-Path $d_{\mathcal{G}}(v_i, v_j)$** | **0.0284 (6.3$\times$ error reduction)** |

---

## 📄 License & Authorship

All mathematical derivations, software implementations, and manuscript texts are the original work of **Muhammed M. El-Sayed** and are distributed under the [MIT License](LICENSE).
