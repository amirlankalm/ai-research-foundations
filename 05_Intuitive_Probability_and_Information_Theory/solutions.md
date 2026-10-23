# 📝 Worked Solutions — Module 05: Intuitive Probability & Information Theory

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises.

---

## Level 1: Recall & Conceptual

### 1. Why MLE Uses Sum of Log Probabilities Instead of Product of Raw Probabilities
1. **Prevents Floating-Point Underflow**: Probabilities are numbers between $0$ and $1$. Multiplying hundreds or thousands of tokens together ($p_1 \times p_2 \times \dots \times p_N$) results in a microscopic number (e.g. $10^{-500}$) that exceeds standard IEEE-754 float precision and rounds directly to `0.0`. Taking the logarithm converts the product into a sum ($\log \prod p_i = \sum \log p_i$), keeping numbers in a well-conditioned range (e.g., $-1520.4$).
2. **Strictly Monotonic**: Because $\log(x)$ is strictly increasing, any parameter configuration that maximizes the log-likelihood also maximizes the original likelihood.
3. **Simplifies Calculus**: Taking the derivative of a sum of terms is vastly simpler than using the product rule on thousands of terms.

### 2. Entropy Calculations
- **100% Probability of Rain**: Entropy is **$0.0$ bits**. $H(P) = - (1.0 \log_2(1.0)) = - 0 = 0$. There is zero uncertainty or surprise.
- **50% Rain / 50% Sun**: Entropy is **$1.0$ bit**. $H(P) = - (0.5 \log_2(0.5) + 0.5 \log_2(0.5)) = - (-0.5 - 0.5) = 1.0$. This is the maximum uncertainty state for a binary variable.

### 3. The Disaster of Removing the KL Penalty in RLHF / GRPO
Without the KL divergence penalty $\beta D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$, the policy experiences **reward hacking** (or *policy drift*). Reward models are imperfect proxy functions. A model trained with pure reinforcement learning quickly finds pathological edge cases (like repeating `<think> answer </think>` 50 times, generating gibberish strings, or using length-bias loopholes) that game the reward score while completely destroying linguistic coherence and instruction-following. The KL penalty tethers the student policy to the original pre-trained foundation model.

---

## Level 2: Hands-On Implementation

### 1. KL Divergence & Asymmetry Demonstration

```python
import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes D_KL(P || Q) = sum(P(x) * log(P(x) / Q(x))).
    Uses small eps to guarantee numerical safety against log(0) and div-by-zero.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

if __name__ == "__main__":
    P = np.array([0.7, 0.2, 0.1])
    Q = np.array([0.3, 0.4, 0.3])
    
    # 1. Identity property: D_KL(P || P) == 0.0
    kl_pp = kl_divergence(P, P)
    print(f"D_KL(P || P) = {kl_pp:.10f} (Expected: 0.0)")
    assert np.isclose(kl_pp, 0.0, atol=1e-7)

    # 2. Asymmetry: D_KL(P || Q) != D_KL(Q || P)
    kl_pq = kl_divergence(P, Q)
    kl_qp = kl_divergence(Q, P)
    print(f"D_KL(P || Q) = {kl_pq:.5f}")
    print(f"D_KL(Q || P) = {kl_qp:.5f}")
    assert not np.isclose(kl_pq, kl_qp)
    print("✅ KL divergence properties verified!")
```

### 2. Softmax & Cross-Entropy on Vocabulary Logits

```python
import numpy as np

def compute_loss(logits: np.ndarray, target_idx: int):
    # 1. Softmax with numerical stability shift
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # 2. Cross-entropy loss = -log(prob of ground-truth target)
    loss = -np.log(probs[target_idx])
    return probs, loss

if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.5, -1.0, 3.5])
    probs, ce_loss = compute_loss(logits, target_idx=4)
    print("Probabilities:", np.round(probs, 4))
    print(f"Probability of Target Token (idx 4): {probs[4]:.4f}")
    print(f"Cross-Entropy Loss: {ce_loss:.4f}")
    
    assert np.isclose(probs.sum(), 1.0)
    assert np.isclose(ce_loss, -np.log(probs[4]))
    print("✅ Softmax and Cross-Entropy loss verified!")
```
