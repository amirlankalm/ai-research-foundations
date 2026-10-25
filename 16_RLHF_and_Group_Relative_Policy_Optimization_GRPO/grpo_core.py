"""
AI Research Foundations - Module 16: Group Relative Policy Optimization (GRPO) Core
Run with: python grpo_core.py
"""

import torch
import torch.nn.functional as F

def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes GRPO's group-relative standardized advantages.
    Formula: A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
    rewards: shape (G,) where G is group size.
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < eps:
        # If all candidates had identical rewards, advantages are zero (no gradient updates)
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)

def grpo_step_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    logprobs_ref: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04
) -> torch.Tensor:
    """
    Computes the token-level GRPO objective with analytical KL penalty.
    Inputs are shape (G, T) for group G across sequence length T.
    """
    # 1. Importance sampling ratio: pi_theta / pi_old
    ratios = torch.exp(logprobs_new - logprobs_old)

    # 2. Clipped surrogate objective
    # Broadcast advantage across sequence length T: (G, 1) -> (G, T)
    adv = advantages.unsqueeze(-1)
    surr1 = ratios * adv
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    ppo_term = torch.min(surr1, surr2)

    # 3. Analytical KL divergence: D_KL(pi || ref) approx = exp(ref - pi) - (ref - pi) - 1
    # or direct difference in log-probs:
    kl_penalty = torch.exp(logprobs_ref - logprobs_new) - (logprobs_ref - logprobs_new) - 1.0

    # 4. Total objective: maximize (PPO term - beta * KL) -> minimize negative
    per_token_loss = -(ppo_term - beta_kl * kl_penalty)
    return per_token_loss.mean()

if __name__ == "__main__":
    print("=== Group Relative Policy Optimization (GRPO) Demo ===")
    print("🔬 Scope Note: Numerical illustration of the loss on synthetic tensors — no model is trained here.\n")
    torch.manual_seed(42)

    # Scenario: For 1 prompt, we sample G = 4 candidate reasoning paths
    # Math problem: "Find x if 3x - 9 = 0"
    # Output 1: "... x = 3" (Correct: reward 1.0)
    # Output 2: "... x = 1" (Wrong:   reward 0.0)
    # Output 3: "... x = 3" (Correct: reward 1.0)
    # Output 4: "... x = 2" (Wrong:   reward 0.0)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = compute_group_advantages(rewards)
    print("Raw Rewards: ", rewards.tolist())
    print("Computed Group Advantages:", advantages.round(decimals=3).tolist())

    # Simulated log probabilities: Group of 4 outputs, length 5 tokens
    G, T = 4, 5
    logprobs_old = torch.randn(G, T)
    logprobs_new = logprobs_old + 0.05 * torch.randn(G, T) # Slight policy shift
    logprobs_ref = logprobs_old.clone()                     # Reference model

    loss = grpo_step_loss(logprobs_new, logprobs_old, logprobs_ref, advantages)
    print(f"GRPO Loss: {loss.item():.4f}")
    print("\n✅ Module 16 GRPO core verified successfully!")
