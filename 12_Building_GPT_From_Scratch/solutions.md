# 📝 Worked Solutions — Module 12: Building GPT from Scratch

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Weight Tying (`wte` and `lm_head`)
In language models, the token embedding matrix `wte` maps discrete token IDs $i \in \{0, \dots, V-1\}$ to continuous representations in $\mathbb{R}^d$, while the final language model head `lm_head` maps continuous hidden states in $\mathbb{R}^d$ back to vocabulary logits in $\mathbb{R}^V$.
- **Mathematical Duality**: If a token's vector representation is close to the hidden state, it should receive high dot-product logit score. Tying their weights ($\mathbf{W}_{\text{head}} = \mathbf{W}_{\text{emb}}^T$) enforces semantic consistency between input representations and output predictions.
- **Parameter Savings**: In large vocabulary models (e.g. $V = 100,000$, $d = 4096$), a single matrix has $409,600,000$ parameters ($\sim 800\text{ MB}$ in fp16). Sharing weights saves hundreds of millions of parameters and prevents the output projection from overfitting.

### 2. Slicing `x[:, [-1], :]` During Generation
During autoregressive generation, we feed the entire running prompt sequence $x_{1:t}$ into the model to compute self-attention across all previous tokens. Because of the causal triangular mask, each position $i$ only attends to positions $\le i$. Therefore, the output vector at the final position $t$ contains the exact contextualized representation predicting the next token $x_{t+1}$. Slicing `x[:, [-1], :]` extracts strictly this terminal token's prediction logits without wasting computation on previous positions that have already been generated.

### 3. Generation Under Extreme Temperatures
- **$\text{Temperature} = 0.0001$**: Scaling logits by $10^{-4}$ dramatically magnifies differences between scores. After softmax, the probability of the single highest-scoring token approaches $1.0$, while all other tokens drop to $0.0$. This effectively reproduces greedy decoding ($\text{argmax}$), producing deterministic, repetitive, and rigid text.
- **$\text{Temperature} = 5.0$**: Dividing logits by 5 flattens score differences, making the softmax distribution approach a uniform distribution over the entire vocabulary. The model samples rare and contextually inappropriate tokens with high probability, producing chaotic hallucinations and gibberish.

---

## Level 2: Hands-On Implementation

### 1. Baseline Verification
Running `python3 12_Building_GPT_From_Scratch/nanogpt_model.py` trains for 150 iterations on CPU:
- Baseline uniform cross-entropy on 30 characters: $-\ln(1/30) \approx 3.4012$.
- Untrained step 0 loss: $\approx 3.66$.
- Step 150 train loss: $2.7260$ ($< 2.80$), and validation loss drops to $3.3321$, decisively beating uniform randomness.

### 2. Top-P (Nucleus) Sampling Implementation

```python
import torch
import torch.nn.functional as F

def sample_top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Applies temperature and Nucleus (Top-p) filtering to logits.
    logits: shape (batch_size, vocab_size)
    """
    # 1. Apply temperature
    logits = logits / max(temperature, 1e-5)
    
    # 2. Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # 3. Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. Remove tokens with cumulative probability above threshold p
    # Shift right by 1 so the first token exceeding p is still kept
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original logit indices and mask out with -infinity
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float("Inf")
    
    # 5. Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# Quick verification
if __name__ == "__main__":
    torch.manual_seed(42)
    test_logits = torch.randn(1, 10)
    token = sample_top_p(test_logits, p=0.8, temperature=0.7)
    print("Sampled token ID:", token.item())
```

---

## Level 3: Research Deep Dive

### 1. Chinchilla Optimal Scaling Laws (Hoffmann et al., 2022)
- **The Chinchilla Ratio**: Hoffmann et al. demonstrated that for compute-optimal training, model size $N$ (non-embedding parameters) and dataset size $D$ (tokens) should scale in equal proportion:
  $$N \propto C^{0.5}, \quad D \propto C^{0.5}$$
  The empirical rule of thumb is approximately **20 tokens per parameter** ($D \approx 20N$).
- **Total Compute Formulation**:
  The total FLOP budget for autoregressive transformer pretraining is approximately:
  $$C \approx 6 N D$$
  *(6 FLOPs per parameter per token: 2 FLOPs for forward pass, 4 FLOPs for backward pass).*
- **Optimal Allocation for $10^{23}$ FLOPs**:
  Setting $D = 20 N$:
  $$C = 6 \cdot N \cdot (20 N) = 120 N^2 = 10^{23}$$
  $$N^2 = \frac{10^{23}}{120} \approx 8.33 \times 10^{20} \implies N \approx 28.8 \times 10^9 \text{ parameters } (\approx 29\text{B})$$
  $$D = 20 \times 28.8 \times 10^9 \approx 576 \times 10^9 \text{ tokens } (\approx 580\text{B tokens})$$
  Rather than training a massive 70B model on only 200B tokens (like original Gopher or Chinchilla baselines), compute is best spent training a ~29B model on ~580B tokens.
