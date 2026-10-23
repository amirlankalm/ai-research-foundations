"""
Comprehensive automated test suite for AI Research Foundations.
Run with: python3 -m pytest tests/ -v
All tests run on CPU and require no external network or paid compute.
"""

import importlib.util
import math
import os
import sys
import numpy as np
import pytest
import torch

def import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Dynamically import modules with numbered directory names
mod01 = import_from_path("mod01", "01_Python_and_Research_Tooling/tooling_demo.py")
mod02 = import_from_path("mod02", "02_NumPy_Vectorization_and_Data/numpy_foundations.py")
mod03 = import_from_path("mod03", "03_Intuitive_Linear_Algebra/linear_algebra_demo.py")
mod04 = import_from_path("mod04", "04_Intuitive_Calculus_and_Optimization/optimization_demo.py")
mod05 = import_from_path("mod05", "05_Intuitive_Probability_and_Information_Theory/probability_info_demo.py")
mod06 = import_from_path("mod06", "06_Classical_Machine_Learning/titanic_baseline.py")
mod07 = import_from_path("mod07", "07_Neural_Networks_and_Autograd_From_Scratch/micrograd_engine.py")
mod09 = import_from_path("mod09", "09_Deep_Learning_Dynamics_and_Architectures/architectures_dynamics.py")
mod10 = import_from_path("mod10", "10_Tokenization_and_Byte_Pair_Encoding/bpe_tokenizer.py")
mod11 = import_from_path("mod11", "11_Attention_and_The_Transformer/attention_mechanisms.py")
mod14 = import_from_path("mod14", "14_Deep_RL_Policy_Gradients_and_PPO/clean_ppo.py")
mod15 = import_from_path("mod15", "15_Post_Training_and_Alignment_SFT_and_DPO/dpo_loss.py")
mod16 = import_from_path("mod16", "16_RLHF_and_Group_Relative_Policy_Optimization_GRPO/grpo_core.py")
mod20 = import_from_path("mod20", "20_Capstone_GRPO_Loss_Walkthrough/reproduce_deepseek_grpo.py")

# -------------------------------------------------------------
# Module 01: Tooling & Batching
# -------------------------------------------------------------
def test_module_01_config():
    cfg = mod01.ExperimentConfig(learning_rate=0.001, batch_size=16)
    assert cfg.learning_rate == 0.001
    assert cfg.batch_size == 16
    d = cfg.to_dict()
    assert d["model_name"] == "nano-transformer"

def test_module_01_batching():
    data = list(range(25))
    batches = list(mod01.batch_generator(data, batch_size=10))
    assert len(batches) == 3
    assert batches[0] == list(range(10))
    assert batches[2] == [20, 21, 22, 23, 24]

# -------------------------------------------------------------
# Module 02: NumPy Vectorization & Softmax
# -------------------------------------------------------------
def test_module_02_softmax():
    logits = np.array([[2.0, 1.0, 0.1], [10.0, 10.0, 10.0]])
    probs = mod02.vectorized_softmax(logits, axis=-1)
    assert np.allclose(probs.sum(axis=-1), [1.0, 1.0])
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]
    assert np.allclose(probs[1], [1/3, 1/3, 1/3])

def test_module_02_cosine_similarity():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, -1.0]])
    sims = mod02.pairwise_cosine_similarity(A, B)
    assert np.isclose(sims[0, 0], 1.0)
    assert np.isclose(sims[0, 1], 0.0)
    assert np.isclose(sims[1, 1], -1.0)

# -------------------------------------------------------------
# Module 03: SVD & LoRA Low-Rank
# -------------------------------------------------------------
def test_module_03_svd():
    np.random.seed(42)
    W = np.random.randn(32, 32)
    W_rank4 = mod03.svd_low_rank_reconstruction(W, rank=4)
    W_rank16 = mod03.svd_low_rank_reconstruction(W, rank=16)
    err4 = np.linalg.norm(W - W_rank4)
    err16 = np.linalg.norm(W - W_rank16)
    assert err16 < err4

# -------------------------------------------------------------
# Module 04: Calculus & Optimizers
# -------------------------------------------------------------
def test_module_04_grad_correctness():
    g = mod04.rosenbrock_grad(1.0, 1.0)
    assert np.allclose(g, [0.0, 0.0])

def test_module_04_adam_convergence():
    init_loss = mod04.rosenbrock(-1.5, 2.0)
    _, adam_loss = mod04.run_optimizer("adam", steps=1000, lr=0.002)
    _, sgd_loss = mod04.run_optimizer("sgd", steps=1000, lr=0.002)
    assert adam_loss < init_loss
    assert adam_loss < sgd_loss

# -------------------------------------------------------------
# Module 05: Probability & Information Theory
# -------------------------------------------------------------
def test_module_05_entropy():
    fair_coin = np.array([0.5, 0.5])
    assert np.isclose(mod05.shannon_entropy(fair_coin, base=2.0), 1.0)
    certain = np.array([1.0, 0.0])
    assert np.isclose(mod05.shannon_entropy(certain, base=2.0), 0.0)

def test_module_05_kl_properties():
    P = np.array([0.7, 0.3])
    Q = np.array([0.4, 0.6])
    assert np.isclose(mod05.kl_divergence(P, P), 0.0)
    assert mod05.kl_divergence(P, Q) > 0.0
    assert not np.isclose(mod05.kl_divergence(P, Q), mod05.kl_divergence(Q, P))

# -------------------------------------------------------------
# Module 06: Classical ML
# -------------------------------------------------------------
def test_module_06_logistic_regression():
    np.random.seed(42)
    X = np.array([[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -3.0]])
    y = np.array([1, 1, 0, 0])
    model = mod06.LogisticRegressionScratch(lr=0.5, n_iters=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)

# -------------------------------------------------------------
# Module 07: Micrograd Autograd
# -------------------------------------------------------------
def test_module_07_autograd():
    Value = mod07.Value
    x = Value(3.0)
    y = Value(-4.0)
    z = (x * y + Value(2.0)).relu()
    z.backward()
    assert z.data == 0.0
    assert x.grad == 0.0
    assert y.grad == 0.0

    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a
    c.backward()
    assert c.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0

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
# Module 10: Tokenization (BPE)
# -------------------------------------------------------------
def test_module_10_bpe_roundtrip():
    text = "Machine learning models rely on tokenization."
    tok = mod10.MiniBPETokenizer()
    tok.train(text, vocab_size=270)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text

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
# Module 16 & 20: GRPO Loss & Math Verifier
# -------------------------------------------------------------
def test_module_16_grpo_advantages():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advs = mod16.compute_group_advantages(rewards)
    assert torch.isclose(advs[0], torch.tensor(1.0))
    assert torch.isclose(advs[1], torch.tensor(-1.0))
    assert torch.isclose(advs.mean(), torch.tensor(0.0), atol=1e-6)

def test_module_20_verifier():
    correct = "<think> 2 + 2 = 4 </think> <answer>4</answer>"
    wrong = "<think> 2 + 2 = 5 </think> <answer>5</answer>"
    assert mod20.verify_math_candidate(correct, "4") == 1.1
    assert mod20.verify_math_candidate(wrong, "4") == 0.1
