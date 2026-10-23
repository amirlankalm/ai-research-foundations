"""
AI Research Foundations - Module 03: Intuitive Linear Algebra & SVD / LoRA Demo
Run with: python linear_algebra_demo.py
"""

import numpy as np

def svd_low_rank_reconstruction(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Reconstructs a matrix using only its top k singular components."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    # Zero out singular values beyond the chosen rank
    S_k = np.zeros_like(S)
    S_k[:rank] = S[:rank]
    reconstructed = np.dot(U, np.dot(np.diag(S_k), Vt))
    return reconstructed

def variance_of_dot_products(dim: int, num_samples: int = 1000) -> float:
    """Measures variance of dot products of random standard normal vectors."""
    q = np.random.randn(num_samples, dim)
    k = np.random.randn(num_samples, dim)
    # Compute dot products along rows
    dots = np.sum(q * k, axis=-1)
    return float(np.var(dots))

if __name__ == "__main__":
    print("=== SVD & Low-Rank Approximation (LoRA Principle) ===")
    np.random.seed(42)
    # Simulate a weight matrix of shape 64 x 64
    W = np.random.randn(64, 64)
    print("Original matrix norm:", np.linalg.norm(W))

    for r in [1, 4, 16, 32, 64]:
        W_approx = svd_low_rank_reconstruction(W, rank=r)
        error = np.linalg.norm(W - W_approx)
        print(f"Rank {r:2d} Reconstruction Error (Frobenius): {error:.4f}")

    print("\n=== Why Transformers Scale by sqrt(d_k) ===")
    for d in [16, 64, 256, 1024]:
        var = variance_of_dot_products(dim=d)
        scaled_var = variance_of_dot_products(dim=d) / d
        print(f"Dim {d:4d} | Raw Dot Product Var: {var:8.2f} (≈ {d}) | Scaled by 1/d Var: {scaled_var:.2f} (≈ 1.0)")

    print("\n✅ Module 03 linear algebra verification complete!")
