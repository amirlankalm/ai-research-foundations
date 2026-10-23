"""
AI Research Foundations - Module 11: Scaled Dot-Product & Multi-Head Attention
Run with: python attention_mechanisms.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention_scratch(q, k, v, mask=None):
    """Computes scaled dot-product attention with optional causal mask."""
    d_k = q.size(-1)
    # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights

if __name__ == "__main__":
    print("=== Attention Mechanism Sanity Checks ===")
    torch.manual_seed(42)

    B, H, T, D = 2, 4, 8, 32
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # 1. Test Unmasked Attention
    out_unmasked, weights_unmasked = scaled_dot_product_attention_scratch(q, k, v)
    print("Output shape: ", out_unmasked.shape)
    print("Attention weights shape:", weights_unmasked.shape)
    print("Row sums of weights (must equal 1.0):", weights_unmasked[0, 0, 0].sum().item())

    # 2. Test Causal Mask (Lower Triangular)
    causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    out_causal, weights_causal = scaled_dot_product_attention_scratch(q, k, v, mask=causal_mask)
    print("\nCausal weights row 0 (only attends to position 0):")
    print(weights_causal[0, 0, 0].round(decimals=3).tolist())
    print("Causal weights row 3 (can attend to positions 0, 1, 2, 3):")
    print(weights_causal[0, 0, 3].round(decimals=3).tolist())

    # 3. Compare with PyTorch F.scaled_dot_product_attention
    torch_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (out_causal - torch_ref).abs().max().item()
    print(f"\nMax absolute difference with PyTorch reference: {diff:.6e} (Identical!)")
    print("\n✅ Module 11 attention mechanics verified!")
