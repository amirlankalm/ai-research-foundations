# 📝 Worked Solutions — Module 09: Deep Learning Dynamics & Architectures

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Kaiming vs. Xavier Variance Factor ($\frac{2}{D_{\text{in}}}$ vs. $\frac{1}{D_{\text{in}}}$)
- Xavier (Glorot) initialization was derived for linear and symmetric saturating activations ($\tanh$, sigmoid) centered around 0.
- ReLU zeros out approximately half of all activations in expectation ($\mathbb{E}[\max(0, z)] = 0$ for half the domain when inputs are centered at 0).
- Halving the active units halves the output activation variance at each layer. To maintain constant variance of activations and gradients throughout deep networks, Kaiming (He) initialization multiplies the variance scale by a factor of 2:
  $$\text{Var}(W) = \frac{2}{D_{\text{in}}}$$

### 2. BatchNorm vs. LayerNorm in Transformers
- **BatchNorm**: Normalizes across the batch dimension $B$ for each individual feature channel. Statistics depend directly on other sequences in the batch and vary wildly with mini-batch size. In variable-length NLP sequences with padding tokens, batch statistics become noisy and distorted. Furthermore, during autoregressive token generation, batch size is often 1, making batch statistics degenerate.
- **LayerNorm**: Normalizes across the hidden feature dimension $D$ independently for each individual token and sequence in the batch. It does not introduce inter-sample dependencies across the batch and works identically regardless of batch size or variable sequence lengths, making it ideal for autoregressive Transformer models.

### 3. Computational Advantage of RMSNorm over LayerNorm
Standard LayerNorm computes both the mean $\mu$ and standard deviation $\sigma$ across the hidden dimension:
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
RMSNorm conjectures that the mean-centering step ($x - \mu$) contributes negligible regularizing benefit. It normalizes strictly by the root mean square:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
By skipping the mean subtraction pass, RMSNorm eliminates redundant GPU memory passes and elementwise operations, reducing execution latency by 10%–50% in memory-bound transformer normalization kernels.

---

## Level 2: Hands-On Implementation

### 1. Custom `RMSNorm` Module in PyTorch

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # x / RMS * gamma
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# Verification
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64)
    rms_norm = RMSNorm(d_model=64)
    out = rms_norm(x)
    print("RMSNorm output shape:", out.shape)
    assert out.shape == x.shape
    # Check that root-mean-square of output across feature dim is approx 1.0
    rms_val = torch.sqrt(torch.mean(out ** 2, dim=-1))
    print("Mean RMS value across tokens:", rms_val.mean().item())
```

### 2. Custom `SwiGLU` (Swish-Gated Linear Unit) Block

```python
class SwiGLU(nn.Module):
    """
    SwiGLU FFN as used in LLaMA, Mistral, and DeepSeek:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(z) = z * sigmoid(z)
        gate = torch.nn.functional.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

# Verification
if __name__ == "__main__":
    swiglu = SwiGLU(d_model=64, hidden_dim=172)
    ffn_out = swiglu(x)
    print("SwiGLU output shape:", ffn_out.shape)
    assert ffn_out.shape == x.shape
```

---

## Level 3: Research Deep Dive

### 1. Pre-LN vs. Post-LN Gradient Propagation
- **Post-LN (Original Transformer, 2017)**:
  $$x_{l+1} = \text{LayerNorm}(x_l + \mathcal{F}(x_l))$$
  The normalization layer wraps the *entire sum*, including the residual path. During backpropagation, the gradient passing through the identity connection is scaled by the Jacobian of the LayerNorm:
  $$\frac{\partial x_{l+1}}{\partial x_l} = \mathcal{J}_{\text{LN}} \cdot (I + \nabla \mathcal{F})$$
  Because the activations grow with depth $L$, the scale factor of LayerNorm causes gradients at the earliest layers to diminish sharply or fluctuate unstably. Post-LN models strictly required a fragile learning rate warmup schedule; otherwise, large initial updates caused gradient explosion or divergence in early training steps.
- **Pre-LN (LLaMA, GPT-3, DeepSeek)**:
  $$x_{l+1} = x_l + \mathcal{F}(\text{LayerNorm}(x_l))$$
  The residual connection is a completely unobstructed additive path:
  $$x_L = x_0 + \sum_{l=0}^{L-1} \mathcal{F}(\text{LayerNorm}(x_l))$$
  The gradient flows directly from output $x_L$ back to input $x_0$ with an unscaled identity term $I$:
  $$\frac{\partial x_L}{\partial x_0} = I + \sum_{l=0}^{L-1} \dots$$
  This ensures stable gradients from step 0, enabling rapid convergence without delicate learning rate warmup gymnastics.
