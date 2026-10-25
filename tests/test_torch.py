"""
AI Research Foundations - Deep Learning & PyTorch Tests
Requires PyTorch (pip install -r requirements-torch.txt).
Automatically skipped if PyTorch is not installed in the environment.
Run with:
    python3 -m pytest tests/test_torch.py -v
"""

import importlib.util
import math
import sys
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

def import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

mod08 = import_from_path("mod08", "08_PyTorch_Deep_Learning_Core/pytorch_pipeline.py")
mod09 = import_from_path("mod09", "09_Deep_Learning_Dynamics_and_Architectures/architectures_dynamics.py")
mod11 = import_from_path("mod11", "11_Attention_and_The_Transformer/attention_mechanisms.py")
mod12 = import_from_path("mod12", "12_Building_GPT_From_Scratch/nanogpt_model.py")
mod14 = import_from_path("mod14", "14_Deep_RL_Policy_Gradients_and_PPO/clean_ppo.py")
mod15 = import_from_path("mod15", "15_Post_Training_and_Alignment_SFT_and_DPO/dpo_loss.py")
mod16 = import_from_path("mod16", "16_RLHF_and_Group_Relative_Policy_Optimization_GRPO/grpo_core.py")
mod18 = import_from_path("mod18", "18_Mechanistic_Interpretability_and_AI_Safety/interpretability_sae_demo.py")
mod20 = import_from_path("mod20", "20_Capstone_GRPO_Loss_Walkthrough/grpo_loss_walkthrough.py")

# -------------------------------------------------------------
# Module 08: PyTorch Pipeline Training Loop & Validation
# -------------------------------------------------------------
def test_module_08_pipeline():
    res = mod08.run_training_pipeline(epochs=5, seed=42, device_name="cpu")
    # Training loss must decrease across epochs
    assert res["final_train_loss"] < res["initial_train_loss"]
    # Model learns synthetic clusters well above 3-class random chance (0.33)
    assert res["val_acc"] > 0.60

# -------------------------------------------------------------
# Module 09: Dynamics, RMSNorm & SwiGLU
# -------------------------------------------------------------
def test_module_09_rmsnorm():
    x = torch.randn(2, 4, 16)
    norm = mod09.RMSNorm(dim=16)
    out = norm(x)
    assert out.shape == (2, 4, 16)
    rms = torch.sqrt(out[0, 0].pow(2).mean())
    assert torch.isclose(rms, torch.tensor(1.0), atol=1e-2)

def test_module_09_swiglu():
    x = torch.randn(2, 4, 16)
    swiglu = mod09.SwiGLUFeedForward(dim=16, hidden_dim=32)
    out = swiglu(x)
    assert out.shape == (2, 4, 16)

# -------------------------------------------------------------
# Module 11: Attention & Causal Masking
# -------------------------------------------------------------
def test_module_11_attention():
    B, H, T, D = 1, 1, 4, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    out, weights = mod11.scaled_dot_product_attention_scratch(q, k, v, mask=mask)
    assert weights[0, 0, 0, 1].item() == 0.0
    assert weights[0, 0, 0, 2].item() == 0.0
    assert torch.isclose(weights.sum(dim=-1), torch.ones(B, H, T)).all()

# -------------------------------------------------------------
# Module 12: Building GPT (nanoGPT) Training & Loss Convergence
# -------------------------------------------------------------
def test_module_12_nanogpt():
    res = mod12.run_worked_experiment(steps=150, verbose=False)
    # Success criterion from README: final loss beats random uniform baseline and drops < 2.80
    assert res["final_train_loss"] < res["baseline_loss"]
    assert res["final_train_loss"] < 2.80
    assert len(res["generated_sample"]) > 0

# -------------------------------------------------------------
# Module 14: PPO Clipped Surrogate
# -------------------------------------------------------------
def test_module_14_ppo_loss():
    logp_new = torch.tensor([-0.5, -1.0])
    logp_old = torch.tensor([-0.7, -0.9])
    adv = torch.tensor([1.0, -1.0])
    loss, ratios = mod14.ppo_clipped_loss(logp_new, logp_old, adv, clip_eps=0.2)
    assert not torch.isnan(loss)
    assert len(ratios) == 2

# -------------------------------------------------------------
# Module 15: DPO Loss
# -------------------------------------------------------------
def test_module_15_dpo_loss():
    ref_w = torch.tensor([-10.0])
    ref_l = torch.tensor([-12.0])
    loss_init, _, _ = mod15.compute_dpo_loss(ref_w, ref_l, ref_w, ref_l, beta=0.1)
    assert torch.isclose(loss_init, torch.tensor(math.log(2.0)), atol=1e-3)

# -------------------------------------------------------------
# Module 16: GRPO Group Advantages
# -------------------------------------------------------------
def test_module_16_grpo_advantages():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advs = mod16.compute_group_advantages(rewards)
    assert torch.isclose(advs[0], torch.tensor(1.0))
    assert torch.isclose(advs[1], torch.tensor(-1.0))
    assert torch.isclose(advs.mean(), torch.tensor(0.0), atol=1e-6)

# -------------------------------------------------------------
# Module 18: Sparse Autoencoder (SAE) Reconstruction & Sparsity
# -------------------------------------------------------------
def test_module_18_sae():
    res = mod18.train_sae_experiment(epochs=100, seed=42, verbose=False)
    # Reconstruction loss must decrease during training
    assert res["final_recon_loss"] < res["init_recon_loss"]
    # L1 sparsity penalty ensures significant latent inactivity
    assert res["final_sparsity"] > 40.0

# -------------------------------------------------------------
# Module 20: Capstone GRPO Math Verifier & Loss Gradient Backprop
# -------------------------------------------------------------
def test_module_20_verifier():
    correct = "<think> 2 + 2 = 4 </think> <answer>4</answer>"
    wrong = "<think> 2 + 2 = 5 </think> <answer>5</answer>"
    # Robust gated verifier (default): wrong candidate receives 0.0 despite think tags (prevents reward hacking)
    assert mod20.verify_math_candidate(correct, "4", strict_format=True) == 1.1
    assert mod20.verify_math_candidate(wrong, "4", strict_format=True) == 0.0
    # Naive un-gated verifier: illustrates reward hacking anti-pattern (+0.1 for tags even when wrong)
    assert mod20.verify_math_candidate(wrong, "4", strict_format=False) == 0.1

def test_module_20_grpo_loss_backward():
    """Verifies that GRPO loss produces finite, correctly shaped gradients that reinforce high advantage."""
    G, T = 4, 8
    rewards = torch.tensor([1.1, 1.1, 0.1, 0.0])
    advantages = mod20.compute_group_advantages(rewards)

    # Standardized advantages must sum/mean to ~0.0
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)

    mask = torch.ones(G, T)
    logprobs_old = torch.randn(G, T)
    logprobs_ref = logprobs_old.clone()

    policy_delta = nn.Parameter(torch.zeros(G, T))
    logprobs_policy = logprobs_old + policy_delta

    loss = mod20.grpo_loss(
        logprobs_policy=logprobs_policy,
        logprobs_old=logprobs_old,
        logprobs_ref=logprobs_ref,
        advantages=advantages,
        mask=mask,
        clip_eps=0.2,
        beta=0.04
    )

    assert not torch.isnan(loss)
    loss.backward()

    # Gradient checks
    assert policy_delta.grad is not None
    assert policy_delta.grad.shape == (G, T)
    assert torch.isfinite(policy_delta.grad).all()

    # In policy gradient minimization (-Adv * ratio), positive advantage candidates have negative loss gradient
    # meaning gradient step (delta - lr * grad) will increase log-prob of high-advantage candidates
    assert policy_delta.grad[0].mean().item() < 0.0
    assert policy_delta.grad[3].mean().item() > 0.0
