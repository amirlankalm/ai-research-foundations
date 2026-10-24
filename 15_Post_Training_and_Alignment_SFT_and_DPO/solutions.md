# 📝 Worked Solutions — Module 15: Post-Training & Alignment (SFT, LoRA & DPO)

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Masking Prompt Tokens with `-100` During SFT
In PyTorch's `nn.CrossEntropyLoss`, `ignore_index=-100` by default.
In conversational SFT, the training data consists of a prompt $x$ and a target assistant response $y$. If we do not mask the prompt tokens, the model computes cross-entropy loss predicting each user token in $x$. This wastes gradient capacity teaching the assistant to predict human questions instead of learning to generate high-quality answers. Setting target labels for all prompt tokens to `-100` ensures that backpropagation occurs exclusively on the assistant's generated response tokens.

### 2. LoRA Initialization: $B = 0$ and $A \sim \mathcal{N}(0, \sigma^2)$
In LoRA, the adapted weight matrix is:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- $A \in \mathbb{R}^{r \times d_{in}}$ is initialized with Gaussian noise $\mathcal{N}(0, 1/r)$.
- $B \in \mathbb{R}^{d_{out} \times r}$ is initialized to all zeros.
Consequently, at initialization ($t = 0$), the product $\Delta W = B \cdot A = 0$.
The adapted model starts with behavior identical to the pretrained base model $W_0$, preventing catastrophic disruption of pretrained capabilities at the beginning of fine-tuning.

### 3. Motivation for DPO Over PPO
Traditional PPO-based RLHF requires running **four large language models concurrently in GPU memory**:
1. Actor (the model being trained).
2. Critic / Value Network (estimating expected return).
3. Reference Model (frozen base model for KL regularization).
4. Reward Model (scoring generated outputs).
Furthermore, PPO requires online rollout generation in the inner loop, which is notoriously unstable and sensitive to hyperparameters. DPO mathematically showed that the Bradley-Terry preference model can be solved analytically, allowing preference optimization directly via an offline classification-style cross-entropy loss with only the Actor and frozen Reference model, eliminating the Critic and online generation entirely.

---

## Level 2: Hands-On Implementation

### 1. Custom `dpo_loss` in PyTorch

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Direct Preference Optimization (DPO) Loss (Rafailov et al., 2023).
    Formula: -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
    """
    # 1. Winning and losing log-ratio margins
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits_margin = beta * (pi_logratios - ref_logratios)

    # 2. Numerically stable loss
    if label_smoothing == 0.0:
        losses = -F.logsigmoid(logits_margin)
    else:
        # Optional conservative label smoothing
        losses = (
            -F.logsigmoid(logits_margin) * (1 - label_smoothing)
            - F.logsigmoid(-logits_margin) * label_smoothing
        )

    return losses.mean()

# Verification
if __name__ == "__main__":
    torch.manual_seed(42)
    # Batch of 4 preference pairs
    pi_w = torch.tensor([-5.2, -4.1, -6.0, -3.8])
    pi_l = torch.tensor([-7.8, -4.9, -6.5, -4.2])
    ref_w = torch.tensor([-6.0, -4.5, -6.2, -4.0])
    ref_l = torch.tensor([-6.5, -4.4, -6.1, -4.1])

    loss = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)
    print("Computed DPO Loss:", loss.item())
    assert loss.item() > 0, "DPO loss must be strictly positive"
```

### 2. Inspecting $\beta$ Sensitivity
- Higher $\beta$ (e.g. $\beta = 0.5$) puts a strong penalty on KL drift from $\pi_{ref}$, keeping responses conservative and close to the base model.
- Lower $\beta$ (e.g. $\beta = 0.01$) allows aggressive policy optimization, which quickly boosts chosen response probabilities but risks mode collapse, verbosity drift, or repetitive loops.

---

## Level 3: Research Deep Dive

### 1. DPO Length Bias and SimPO / IPO Solutions
- **How Length Bias Arises**: The sequence log probability is a sum of per-token log probabilities:
  $$\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$$
  Because log probabilities are negative, longer sequences generally sum to more negative numbers. However, when a model increases the per-token probability of a response by even a small delta $\epsilon$, accumulating that delta over 500 tokens produces a $500\epsilon$ boost, compared to only $50\epsilon$ for a concise 50-token response. The optimizer rapidly discovers that generating long, verbose explanations yields huge reward margins.
- **SimPO Solution (Meng et al., 2024)**:
  SimPO replaces the raw sum of log probabilities with **length-normalized average log probability**:
  $$p_{\text{SimPO}}(y \mid x) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$
  It also introduces a positive target margin $\gamma$:
  $$\mathcal{L}_{\text{SimPO}} = -\log \sigma \left( \frac{\beta}{|y_w|} \log \pi(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi(y_l \mid x) - \gamma \right)$$
  This completely removes the need for a reference model $\pi_{ref}$ while eliminating length exploitation.
