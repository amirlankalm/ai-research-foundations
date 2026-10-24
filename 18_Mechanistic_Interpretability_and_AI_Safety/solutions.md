# 📝 Worked Solutions — Module 18: Mechanistic Interpretability & AI Safety

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Q-K Circuit vs. O-V Circuit in Attention Heads
Anthropic's *Mathematical Framework for Transformer Circuits* (Elhage et al., 2021) shows that an attention head decomposes into two distinct sub-circuits:
- **The Q-K Circuit ($W_Q^T W_K$)**: Determines **where** to move information. It compares query vectors from destination tokens with key vectors from source tokens to compute the scalar attention score matrix $A = \text{softmax}(x W_Q W_K^T x^T / \sqrt{d})$.
- **The O-V Circuit ($W_V W_O$)**: Determines **what** information to move. It maps source token representations into a subspace via $W_V$ and projects them back into the residual stream via $W_O$, independently of where the information was routed.

### 2. Induction Heads & In-Context Learning
- **Pattern Detected**: An induction head detects sequences of the form $[A][B] \dots [A] \to [B]$. If token $[A]$ appeared previously followed by $[B]$, when $[A]$ appears again later in the context, the induction head attends back to the previous $[A]$ (or its successor) and predicts that $[B]$ will follow again.
- **Two-Layer Mechanism**: Layer 1 has a previous-token head that copies info from token $i$ to position $i+1$. Layer 2's induction head searches the context for tokens whose Layer 1 representation matches the current token, shifting probability mass toward the token that historically followed it.
- This simple mechanical circuit explains how LLMs perform in-context few-shot learning and pattern copying without weight updates.

### 3. Polysemanticity & The Superposition Hypothesis
- **Polysemanticity**: Individual neurons in a deep neural network frequently fire for multiple completely unrelated concepts (e.g., a single neuron firing for academic citations, Arabic script, and Korean pop lyrics).
- **The Superposition Hypothesis**: The real world contains vastly more independent conceptual features than the number of physical neurons/dimensions $D$ available in the model's hidden representation. Because concepts are sparse (most concepts are absent in any given token), models pack $M \gg D$ quasi-orthogonal feature directions into $D$-dimensional space. Projecting these features onto standard coordinate axes makes individual neurons appear polysemantic.

---

## Level 2: Hands-On Implementation

### 1. Custom `SparseAutoencoder` (SAE) Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, dict_size: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.l1_coeff = l1_coeff

        # Encoder projects d_model -> expanded dictionary size
        self.encoder = nn.Linear(d_model, dict_size)
        # Decoder projects dictionary size -> d_model
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

        # Normalize decoder dictionary columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor):
        # 1. Sparse feature activations via ReLU
        feature_acts = F.relu(self.encoder(x))
        
        # 2. Reconstructed activation vector
        reconstructed = self.decoder(feature_acts)
        
        # 3. Losses
        recon_loss = F.mse_loss(reconstructed, x)
        l1_loss = self.l1_coeff * feature_acts.sum(dim=-1).mean()
        total_loss = recon_loss + l1_loss

        return {
            "reconstructed": reconstructed,
            "feature_acts": feature_acts,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss
        }

# Verification
if __name__ == "__main__":
    torch.manual_seed(42)
    sae = SparseAutoencoder(d_model=64, dict_size=256, l1_coeff=0.01)
    mock_activations = torch.randn(32, 64)
    out = sae(mock_activations)
    print("Total Loss:", out["loss"].item())
    print("Reconstruction MSE:", out["recon_loss"].item())
    print("L1 Sparsity Loss:  ", out["l1_loss"].item())
    sparsity = (out["feature_acts"] > 0).float().mean().item()
    print(f"Active Features: {sparsity * 100:.1f}% of dictionary")
```

---

## Level 3: Research Deep Dive

### 1. Activation Steering with SAE Features
- **Mechanism**: Once an SAE feature direction $\vec{d}_i \in \mathbb{R}^d$ is mapped to a human concept (e.g. "Golden Gate Bridge" or "Deceptive Reasoning"), researchers can steer model behavior at runtime by adding the vector directly into intermediate residual activations:
  $$x_{\text{steered}} = x + \alpha \cdot \vec{d}_i$$
  - Clamping $\alpha > 0$ forces the model to fixate on the target concept regardless of the user prompt.
  - Setting $\alpha < 0$ suppresses the concept (e.g., dampening sycophancy or toxic styles).
- **Limitations Compared to RLHF**:
  1. **Locality & Distributional Drift**: Intervening directly in the activation space shifts activations outside the natural manifold the downstream transformer layers were trained on, which can degrade coherence or induce grammatical decay.
  2. **Layer Sensitivity**: Steered concepts must be injected at specific layers (typically middle to late residual streams) and may not generalize across different tasks or multi-turn dialogues.
  3. **Non-Permanent**: Steering requires real-time forward pass hooks and inference modifications rather than baked-in post-training parameter weights.
