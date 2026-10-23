"""
AI Research Foundations - Core Tests (NumPy, SciPy, Python Standard Library)
Zero PyTorch dependencies. Can be run with:
    python3 -m pytest tests/test_core.py -v
"""

import importlib.util
import math
import sys
import numpy as np
import pytest

def import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

mod01 = import_from_path("mod01", "01_Python_and_Research_Tooling/tooling_demo.py")
mod02 = import_from_path("mod02", "02_NumPy_Vectorization_and_Data/numpy_foundations.py")
mod03 = import_from_path("mod03", "03_Intuitive_Linear_Algebra/linear_algebra_demo.py")
mod04 = import_from_path("mod04", "04_Intuitive_Calculus_and_Optimization/optimization_demo.py")
mod05 = import_from_path("mod05", "05_Intuitive_Probability_and_Information_Theory/probability_info_demo.py")
mod06 = import_from_path("mod06", "06_Classical_Machine_Learning/titanic_baseline.py")
mod07 = import_from_path("mod07", "07_Neural_Networks_and_Autograd_From_Scratch/micrograd_engine.py")
mod10 = import_from_path("mod10", "10_Tokenization_and_Byte_Pair_Encoding/bpe_tokenizer.py")

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
# Module 10: Tokenization (BPE)
# -------------------------------------------------------------
def test_module_10_bpe_roundtrip():
    text = "Machine learning models rely on tokenization."
    tok = mod10.MiniBPETokenizer()
    tok.train(text, vocab_size=270)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
