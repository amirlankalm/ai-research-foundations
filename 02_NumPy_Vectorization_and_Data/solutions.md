# 📝 Worked Solutions — Module 02: NumPy, Vectorization & Data

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises.

---

## Level 1: Recall & Conceptual

### 1. Definitions
- **Contiguous memory**: An arrangement where all elements of an array sit adjacent to one another in physical RAM. This enables the hardware CPU to pull sequential chunks into high-speed L1/L2 cache lines (64 bytes at a time) and execute SIMD instructions, making array iteration orders of magnitude faster than following scattered pointer addresses (like Python lists).
- **Axis**: A specific direction along which an array is indexed. In a 2D matrix, `axis 0` runs vertically across rows (collapsing rows when aggregated), and `axis 1` runs horizontally across columns.
- **Broadcasting Conditions**: Starting from trailing (rightmost) dimensions and moving backwards, two dimensions are compatible if:
  1. They are equal, or
  2. One of them is 1.

### 2. Dimension Compatibility Check
For shapes `(16, 1, 64)` and `(16, 32, 1)`:
- Trailing dimension: `64` vs `1` $\to$ compatible (expands to `64`).
- Middle dimension: `1` vs `32` $\to$ compatible (expands to `32`).
- Leading dimension: `16` vs `16` $\to$ compatible (matches `16`).
- **Result**: The element-wise operation succeeds with output shape **`(16, 32, 64)`**.

---

## Level 2: Hands-On Implementation

### 1. Vectorized Cosine Similarity Matrix

```python
import numpy as np

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes pairwise cosine similarity between A of shape (N, D) and B of shape (M, D).
    Output shape: (N, M).
    Purely vectorized in NumPy with zero Python loops.
    """
    # 1. Dot product numerator: (N, D) @ (D, M) -> (N, M)
    dot_products = A @ B.T

    # 2. L2 norms: (N, 1) and (1, M)
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)  # Shape (N, 1)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True).T # Shape (1, M)

    # 3. Broadcasting division
    sim = dot_products / (norm_A * norm_B + eps)
    return sim

# Test sanity
if __name__ == "__main__":
    A = np.random.randn(5, 32)
    B = np.random.randn(10, 32)
    sims = cosine_similarity_matrix(A, B)
    assert sims.shape == (5, 10)
    assert np.all(sims >= -1.0) and np.all(sims <= 1.0)
    print("✅ Pairwise cosine similarity shape:", sims.shape)
```

### 2. Causal Masked Softmax on Attention Scores

```python
import numpy as np

def causal_softmax(scores: np.ndarray) -> np.ndarray:
    """
    Applies causal masking (upper triangle set to -inf) and row-wise softmax.
    """
    N = scores.shape[0]
    # Causal mask: True where col > row (upper triangle without diagonal)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    masked_scores = scores.copy()
    masked_scores[mask] = -np.inf

    # Numerically stable softmax: subtract row-wise max
    # Note: where an entire row is -inf, max is -inf; here diagonal is finite so max is valid
    shift_scores = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shift_scores)
    # Masked positions had -inf, exp(-inf) = 0.0
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs

if __name__ == "__main__":
    raw_scores = np.random.randn(8, 8)
    attn_weights = causal_softmax(raw_scores)
    print("Row sums (must be 1.0):", attn_weights.sum(axis=-1))
    print("Upper-triangular entries (must be 0.0):", attn_weights[0, 1:])
    assert np.allclose(attn_weights.sum(axis=-1), 1.0)
    assert np.all(attn_weights[np.triu(np.ones((8, 8), dtype=bool), k=1)] == 0.0)
    print("✅ Causal softmax verified!")
```
