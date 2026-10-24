# 📝 Worked Solutions — Module 14: Deep RL: Policy Gradients & PPO

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. The Log-Derivative Trick
The expectation in policy gradients is taken with respect to the policy's own action distribution: $\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. Because the sampling distribution itself depends on $\theta$, we cannot simply push the gradient inside the expectation directly:
$$\nabla_\theta \sum_a \pi_\theta(a \mid s) Q(s, a) = \sum_a \nabla_\theta \pi_\theta(a \mid s) Q(s, a)$$
Using the identity $\nabla_\theta \log \pi_\theta(a \mid s) = \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}$, we rewrite:
$$\sum_a \pi_\theta(a \mid s) \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q(s, a) \right] = \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q(s, a) \right]$$
This transforms the gradient into an expectation that can be approximated via standard Monte Carlo batch sampling!

### 2. Role of the Critic & Variance Reduction
The **Critic** learns to approximate the state value baseline $V_\phi(s) \approx \mathbb{E}[R \mid s]$. In the Actor update, instead of using raw cumulative return $G_t$ *(which varies wildly depending on random future dice rolls)*, we subtract the baseline to compute the advantage:
$$A(s, a) = Q(s, a) - V(s)$$
Subtracting a state-dependent baseline does not introduce bias into the expectation, but it drastically reduces variance because the policy only learns from whether an action was *better or worse than expected* for that state.

### 3. Why PPO Clips the Probability Ratio $r_t(\theta)$
In vanilla policy gradients, taking a gradient step that changes policy weights too much causes catastrophic collapse: the new policy becomes terrible, and because future data is sampled from this broken policy, it can never recover.
PPO defines the importance ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and clips it to $[1 - \epsilon, 1 + \epsilon]$ (typically $[0.8, 1.2]$):
$$L^{\text{CLIP}} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\hat{A}_t)$$
If an update tries to push the ratio beyond $1.2$ for positive advantage or below $0.8$ for negative advantage, the gradient is zeroed out, preventing destructive policy updates.

---

## Level 2: Hands-On Implementation

### 1. PPO Clipped Surrogate Objective in PyTorch

```python
import torch

def ppo_clip_objective(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2
) -> torch.Tensor:
    """
    Computes PPO Clipped Surrogate Loss:
    L = -mean(min(r * A, clip(r, 1 - eps, 1 + eps) * A))
    """
    # 1. Compute probability ratio r_t = exp(logp_new - logp_old)
    ratios = torch.exp(logprobs_new - logprobs_old)

    # 2. Unclipped surrogate
    surr1 = ratios * advantages

    # 3. Clipped surrogate
    ratios_clipped = torch.clamp(ratios, 1.0 - eps, 1.0 + eps)
    surr2 = ratios_clipped * advantages

    # 4. Pessimistic minimum (take negative for gradient descent minimization)
    loss = -torch.min(surr1, surr2).mean()
    return loss, ratios

if __name__ == "__main__":
    logp_new = torch.tensor([-0.5, -1.2, -0.3])
    logp_old = torch.tensor([-0.6, -1.0, -0.3])
    adv = torch.tensor([1.5, -0.8, 0.0])

    loss, ratios = ppo_clip_objective(logp_new, logp_old, adv, eps=0.2)
    print(f"Computed Ratios: {ratios.tolist()}")
    print(f"PPO Clipped Policy Loss: {loss.item():.5f}")
    assert not torch.isnan(loss)
    print("✅ PPO clipped objective verified!")
```

---

## Level 3: Research Deep Dive (Approach & Conceptual Guidance)

### Generalized Advantage Estimation (GAE)
- **The Trade-off**:
  - **TD(0) ($\lambda = 0$)**: Advantage is $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$. Low variance because it only relies on one step of reward noise, but high bias because it relies heavily on the learned critic $V(s)$.
  - **Monte Carlo ($\lambda = 1$)**: Advantage is $\sum_{k=0}^\infty \gamma^k r_{t+k} - V(s_t)$. Zero bias (true returns), but extremely high variance due to compounding stochastic transitions and actions along the entire trajectory.
- **GAE($\gamma, \lambda$)**: Exponentially weights $k$-step temporal difference residuals by $\lambda^k$. Setting $\lambda \approx 0.95$ yields the optimal sweet spot, retaining the stability of TD updates while incorporating multi-step return signals.
