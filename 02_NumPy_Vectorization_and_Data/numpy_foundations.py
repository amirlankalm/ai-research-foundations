"""
AI Research Foundations - Module 02: NumPy Vectorization & Broadcasting
Run with: python numpy_foundations.py
"""

import numpy as np

def vectorized_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    Subtracts maximum value along the axis to prevent numerical overflow in exp().
    """
    # max value for numerical stability (prevents e^(1000) = inf)
    max_vals = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_vals)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def pairwise_cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between all pairs of rows in A (N, D) and B (M, D).
    Returns matrix of shape (N, M).
    """
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)  # (N, 1)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)  # (M, 1)
    
    A_normalized = A / (norm_A + 1e-9)
    B_normalized = B / (norm_B + 1e-9)
    
    return np.matmul(A_normalized, B_normalized.T)

if __name__ == "__main__":
    print("=== Broadcasting Demo ===")
    token_features = np.ones((2, 4, 3))  # 2 batches, 4 tokens, 3 features
    bias = np.array([10.0, 20.0, 30.0])  # 3 features
    output = token_features + bias
    print("Tokens Shape:", token_features.shape)
    print("Bias Shape:", bias.shape)
    print("Output Shape:", output.shape)
    print("Sample Output batch 0:\n", output[0])

    print("\n=== Softmax Verification ===")
    test_logits = np.array([[2.0, 1.0, 0.1], [1.0, 5.0, 2.0]])
    probs = vectorized_softmax(test_logits, axis=-1)
    print("Softmax Probs:\n", probs)
    print("Row Sums (Should be 1.0):", np.sum(probs, axis=-1))

    print("\n=== Pairwise Cosine Similarity ===")
    embeddings1 = np.random.randn(3, 16)
    embeddings2 = np.random.randn(5, 16)
    sims = pairwise_cosine_similarity(embeddings1, embeddings2)
    print("Similarity Matrix Shape:", sims.shape)
    print("Similarity range: min =", sims.min(), "max =", sims.max())
    print("\n✅ Module 02 demo executed successfully!")
