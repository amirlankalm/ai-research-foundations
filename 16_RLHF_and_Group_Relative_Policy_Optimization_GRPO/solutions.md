# 📝 Worked Solutions — Module 16: RLHF & Group Relative Policy Optimization (GRPO)

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Why Traditional PPO Requires a Critic Network
In PPO, computing the policy gradient requires an advantage estimate $A(s, a) = Q(s, a) - V(s)$. The Critic is a second neural network (often as large as the policy model itself) trained via mean squared error to predict the expected value $V(s)$ from each state. Without the Critic, using raw episode rewards produces massive variance that destabilizes training.

### 2. How GRPO Computes the Baseline Without a Critic
For each input prompt $q$, GRPO samples a **group of $G$ independent completions** $\{o_1, o_2, \dots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$. It evaluates the reward $r_i$ for each completion using a verifier or reward model, and calculates the baseline dynamically as the **mean of the group**:
$$\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i$$
The advantage of completion $i$ is simply how much better it performed compared to the rest of the group: $\hat{A}_i = \frac{r_i - \bar{r}}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$.

### 3. Dividing by Standard Deviation & Case When All Rewards Are Equal
- **Dividing by $\text{std}(r) + \epsilon$**: Standardizes the advantages to have unit variance ($\sigma \approx 1.0$), ensuring that the magnitude of policy updates remains consistent across easy questions (where rewards have low spread) and hard questions (where rewards have high spread).
- **When All Outputs Receive the Same Reward** (e.g., all correct `[1, 1, 1, 1]` or all incorrect `[0, 0, 0, 0]`): The sample variance is $0.0$. In this case, $\hat{A}_i = 0.0$ for all outputs. This is desirable: if all sampled answers were identical in quality, none was relatively better or worse than the others, so the policy receives zero gradient and leaves its parameters untouched.

---

## Level 2: Hands-On Implementation

### 1. `compute_group_advantages` in PyTorch

```python
import torch

def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes group-relative standardized advantages:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    If std(r) < eps (all rewards identical), returns zeros.
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    
    if std < eps:
        return torch.zeros_like(rewards)
    
    return (rewards - mean) / (std + eps)

if __name__ == "__main__":
    # Test case 1: alternating rewards
    r1 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    a1 = compute_group_advantages(r1)
    print("Test 1 Advantages:", a1.tolist())
    assert torch.allclose(a1, torch.tensor([1.0, -1.0, 1.0, -1.0]))

    # Test case 2: identical rewards
    r2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    a2 = compute_group_advantages(r2)
    print("Test 2 Advantages:", a2.tolist())
    assert torch.allclose(a2, torch.zeros(4))
    print("✅ compute_group_advantages verified!")
```

---

## Level 3: Research Deep Dive (Approach & Conceptual Guidance)

### GRPO vs. Leave-One-Out Baseline
- **Leave-One-Out (LOO)**: Sets $b_i = \frac{1}{G-1} \sum_{j \neq i} r_j$. Because $r_i$ is completely excluded from its own baseline, LOO is mathematically strictly unbiased: $\mathbb{E}[b_i \nabla \log \pi(o_i)] = 0$.
- **GRPO Group Standardization**: Includes $r_i$ in the group mean and divides by group standard deviation. While this introduces a tiny finite-sample bias of order $O(1/G)$, it drastically stabilizes training variance across varying difficulty prompts.
- **Binary Rewards in Math**: In mathematical reasoning benchmarks like GSM8K or MATH, rewards are discrete $r \in \{0, 1\}$. If 1 out of 8 candidates finds the right proof ($r = [1, 0, 0, 0, 0, 0, 0, 0]$), group standardization assigns a massive positive advantage ($\approx +2.6$) to the single working solution, concentrating the gradient signal on the successful reasoning trajectory.
