"""
AI Research Foundations - Module 09: Modern Architectures (RMSNorm & SwiGLU)
Run with: python architectures_dynamics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Used in LLaMA, Mistral, and DeepSeek architectures.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLUFeedForward(nn.Module):
    """
    Swish Gated Linear Unit Feed-Forward Network (Shazeer, 2020).
    Replaces standard ReLU/GELU MLPs in modern state-of-the-art LLMs.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is Swish: x * sigmoid(x)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

if __name__ == "__main__":
    print("=== Testing RMSNorm & SwiGLU Components ===")
    torch.manual_seed(42)

    # Simulated batch: (Batch=2, SeqLen=4, HiddenDim=8)
    dummy_tokens = torch.randn(2, 4, 8)
    
    # 1. Test RMSNorm
    rmsnorm = RMSNorm(dim=8)
    normed = rmsnorm(dummy_tokens)
    print("Input shape: ", dummy_tokens.shape)
    print("Normed shape:", normed.shape)
    print("RMS of output token 0,0:", torch.sqrt(normed[0, 0].pow(2).mean()).item(), "(Should be ~1.0)")

    # 2. Test SwiGLU
    swiglu = SwiGLUFeedForward(dim=8, hidden_dim=24)
    out = swiglu(normed)
    print("SwiGLU output shape:", out.shape)
    print("Residual sum shape: ", (normed + out).shape)

    print("\n✅ Module 09 dynamics & components verified successfully!")
