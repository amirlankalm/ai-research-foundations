# 📝 Worked Solutions — Module 03: Intuitive Linear Algebra

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises.

---

## Level 1: Recall & Conceptual

### 1. Geometric Relationship when $\vec{u} \cdot \vec{v} = 0$
The vectors are **orthogonal** (perpendicular, forming a 90° angle). Because $\vec{u} \cdot \vec{v} = \|\vec{u}\| \|\vec{v}\| \cos(\theta)$, a dot product of zero means $\cos(\theta) = 0$, implying zero projection of one vector along the other.

### 2. Implication of $\det(A) = 0$
The determinant measures how much a transformation scales areas or volumes. If $\det(A) = 0$, the transformation collapses space into a lower dimension (e.g., squashing a 2D plane onto a 1D line or a 3D volume into a flat sheet). Because infinite points collapse onto the same line or point, information is permanently lost and the operation **cannot be inverted** ($A^{-1}$ does not exist).

### 3. Plain-English Definition & Analogy of an Eigenvector
An **eigenvector** of a transformation is a special direction in space that **does not rotate or change its tilt** when the transformation is applied; it only gets stretched, shrunk, or flipped by a scalar factor called the **eigenvalue** ($\lambda$).
- *Analogy*: Imagine stretching a sheet of rubber diagonally. Most lines drawn on the rubber sheet will bend or change their angle. But lines drawn exactly along the direction of the stretch stay completely straight—they simply stretch longer. Those special unchanged axes are the eigenvectors.

---

## Level 2: Hands-On Implementation

### 1 & 2. SVD Low-Rank Approximation & Reconstruction Error

```python
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approx(W: np.ndarray, rank: int) -> np.ndarray:
    """
    Computes rank-k approximation using Singular Value Decomposition:
    W = U @ S @ Vh  ->  W_approx = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]
    """
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    Vh_k = Vh[:rank, :]
    return U_k @ S_k @ Vh_k

def frobenius_error(W: np.ndarray, W_hat: np.ndarray) -> float:
    """Computes Frobenius norm ||W - W_hat||_F."""
    return float(np.linalg.norm(W - W_hat, ord="fro"))

if __name__ == "__main__":
    np.random.seed(42)
    N = 64
    W = np.random.randn(N, N)
    
    ranks = [1, 2, 4, 8, 16, 32, 64]
    errors = []
    
    print("=== SVD Low-Rank Reconstruction on (64x64) Matrix ===")
    for r in ranks:
        W_approx = low_rank_approx(W, rank=r)
        err = frobenius_error(W, W_approx)
        errors.append(err)
        print(f"Rank {r:2d} | Frobenius Error: {err:6.3f} | Fraction of Original Energy Lost: {err / np.linalg.norm(W):.2%}")

    assert errors[0] > errors[1] > errors[2] > errors[3] > errors[4]
    assert np.isclose(errors[-1], 0.0, atol=1e-10)
    print("✅ All low-rank approximations verified!")
```
