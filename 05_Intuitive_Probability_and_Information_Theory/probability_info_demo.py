"""
AI Research Foundations - Module 05: Entropy, Cross-Entropy & KL Divergence Demo
Run with: python probability_info_demo.py
"""

import numpy as np

def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Computes Shannon entropy H(P) in bits (base 2) or nats (base e)."""
    p = p[p > 0]  # Avoid log(0)
    return float(-np.sum(p * (np.log(p) / np.log(base))))

def cross_entropy_loss(logits: np.ndarray, target_idx: int) -> float:
    """Computes cross-entropy loss from raw logits for a single target token."""
    # Numerically stable softmax
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    return float(-np.log(probs[target_idx] + 1e-12))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Kullback-Leibler Divergence D_KL(P || Q).
    P is the reference distribution, Q is the approximating distribution.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

if __name__ == "__main__":
    print("=== Entropy Analysis ===")
    uniform_p = np.array([0.25, 0.25, 0.25, 0.25])
    confident_p = np.array([0.94, 0.02, 0.02, 0.02])
    print(f"Entropy of Uniform distribution (max uncertainty): {shannon_entropy(uniform_p):.3f} bits")
    print(f"Entropy of Confident distribution (low uncertainty): {shannon_entropy(confident_p):.3f} bits")

    print("\n=== Cross-Entropy Loss ===")
    sample_logits = np.array([1.2, 0.5, 3.8, -0.4])
    loss_correct = cross_entropy_loss(sample_logits, target_idx=2)  # Index 2 had highest logit (3.8)
    loss_wrong = cross_entropy_loss(sample_logits, target_idx=3)    # Index 3 had lowest logit (-0.4)
    print(f"Cross-entropy loss when predicting correctly (idx 2): {loss_correct:.4f}")
    print(f"Cross-entropy loss when predicting poorly    (idx 3): {loss_wrong:.4f}")

    print("\n=== KL Divergence Asymmetry (RLHF Regularizer) ===")
    P = np.array([0.7, 0.2, 0.1])
    Q = np.array([0.4, 0.4, 0.2])
    kl_pq = kl_divergence(P, Q)
    kl_qp = kl_divergence(Q, P)
    print(f"D_KL(P || Q): {kl_pq:.4f}")
    print(f"D_KL(Q || P): {kl_qp:.4f}")
    print(f"Are they equal? {np.isclose(kl_pq, kl_qp)} (KL is asymmetric!)")
    print("\n✅ Module 05 probability & information theory demo complete!")
