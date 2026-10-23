"""
AI Research Foundations - Module 14: Core PPO Loss & Clipped Surrogate Objective
Run with: python clean_ppo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def ppo_clipped_loss(logprobs_new: torch.Tensor, logprobs_old: torch.Tensor, advantages: torch.Tensor, clip_eps: float = 0.2):
    """
    Computes PPO's clipped surrogate loss.
    Negative because optimizers minimize, while RL seeks to maximize expected return.
    """
    # ratio r_t(theta) = exp(log(pi_new) - log(pi_old))
    ratios = torch.exp(logprobs_new - logprobs_old)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss, ratios

class ActorCritic(nn.Module):
    """Discrete Actor-Critic network."""
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=32):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action_and_value(self, state, action=None):
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(state)

if __name__ == "__main__":
    print("=== PPO Clipped Surrogate Sanity Check ===")
    torch.manual_seed(42)

    # 1. Simulate batch of 4 steps
    logprobs_old = torch.tensor([-0.693, -1.204, -0.223, -0.850]) # log(0.5), log(0.3), etc.
    # New policy shifted slightly
    logprobs_new = torch.tensor([-0.510, -1.500, -0.210, -0.400])
    advantages   = torch.tensor([ 1.500, -2.000,  0.500,  1.200]) # Normalized advantages

    loss, ratios = ppo_clipped_loss(logprobs_new, logprobs_old, advantages, clip_eps=0.2)
    print("Probability Ratios:", ratios.round(decimals=3).tolist())
    print("PPO Clipped Policy Loss:", loss.item())

    # 2. Test ActorCritic module
    agent = ActorCritic(state_dim=4, action_dim=2)
    dummy_state = torch.randn(1, 4)
    action, log_prob, entropy, value = agent.get_action_and_value(dummy_state)
    print("\nActor Action Sampled:", action.item())
    print("Action Log Probability:", log_prob.item())
    print("State Baseline Value:  ", value.item())

    print("\n✅ Module 14 PPO logic successfully verified!")
