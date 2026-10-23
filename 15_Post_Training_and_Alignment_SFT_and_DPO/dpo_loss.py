"""
AI Research Foundations - Module 15: Direct Preference Optimization (DPO) Loss Implementation
Run with: python dpo_loss.py
"""

import torch
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Direct Preference Optimization (DPO) loss.
    All inputs are sequence-level summed log probabilities: Shape (Batch,)
    """
    # 1. Compute log ratios between active policy and reference model
    pi_logratios_w = policy_chosen_logps - reference_chosen_logps
    pi_logratios_l = policy_rejected_logps - reference_rejected_logps

    # 2. Compute preference margin
    logits = beta * (pi_logratios_w - pi_logratios_l)

    # 3. DPO Loss: -E[log sigma(beta * (log(pi_w/ref_w) - log(pi_l/ref_l)))]
    losses = -F.logsigmoid(logits)

    # 4. Implicit rewards for tracking and logging
    chosen_rewards = beta * pi_logratios_w.detach()
    rejected_rewards = beta * pi_logratios_l.detach()

    return losses.mean(), chosen_rewards, rejected_rewards

if __name__ == "__main__":
    print("=== Direct Preference Optimization (DPO) Demo ===")
    torch.manual_seed(42)

    # Simulated batch of 2 preference pairs
    # Sequence log probabilities under reference model (frozen)
    ref_chosen_logps   = torch.tensor([-12.5, -8.3])
    ref_rejected_logps = torch.tensor([-14.1, -9.0])

    # Initial active policy: initially identical to reference
    pol_chosen_logps   = torch.tensor([-12.5, -8.3], requires_grad=True)
    pol_rejected_logps = torch.tensor([-14.1, -9.0], requires_grad=True)

    # Case 1: Initial state (identical to ref) -> loss = -log(sigmoid(0)) = -log(0.5) ≈ 0.693
    loss_init, r_w, r_l = compute_dpo_loss(
        pol_chosen_logps, pol_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )
    print(f"Initial DPO Loss (Identical models): {loss_init.item():.4f} (Expected ≈ 0.6931)")

    # Case 2: After training: model boosts winning tokens and suppresses losing tokens
    pol_chosen_improved   = torch.tensor([-10.2, -6.1]) # Higher probability (less negative)
    pol_rejected_improved = torch.tensor([-18.0, -12.5]) # Lower probability (more negative)

    loss_improved, r_w_imp, r_l_imp = compute_dpo_loss(
        pol_chosen_improved, pol_rejected_improved,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )
    print(f"Improved DPO Loss: {loss_improved.item():.4f} (Significantly lower!)")
    print(f"Winning Rewards:  {r_w_imp.round(decimals=3).tolist()}")
    print(f"Losing Rewards:   {r_l_imp.round(decimals=3).tolist()}")

    print("\n✅ Module 15 DPO loss verification complete!")
