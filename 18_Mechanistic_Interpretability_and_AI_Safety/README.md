# 18. Mechanistic Interpretability & AI Safety

<p align="center">
  <img src="../assets/18_interpretability_sae.png" alt="18_Mechanistic_Interpretability_and_AI_Safety Diagram" width="85%">
</p>

> [!NOTE]
> ⏱️ **Estimated Study Time**: ~3 hours reading + ~2 hours exercises  
> 🎯 **What "Done" Looks Like**:
> - [ ] You can explain polysemanticity, superposition, and why Sparse Autoencoders (SAEs) disentangle monosemantic features.
> - [ ] You understand how induction heads implement in-context few-shot learning across two attention layers.
> - [ ] You ran `python3 18_Mechanistic_Interpretability_and_AI_Safety/interpretability_sae_demo.py` and saw L1 sparsity enforce clean feature separation.
> - [ ] You attempted the Level 1 & Level 2 exercises.


Welcome to Module 18 of **AI Research Foundations**. Deep neural networks are often described as "black boxes." But in frontier AI research, treating models as inscrutable mystery boxes is dangerous. **Mechanistic Interpretability** treats neural networks like compiled machine code: our job is to reverse-engineer the weights back into human-understandable algorithms and verify their safety.

---

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the goals of **Mechanistic Interpretability** *(reverse-engineering neural circuits)*.
- Deconstruct the two sub-circuits inside every attention head: the **Q-K Circuit** and the **O-V Circuit**.
- Discover **Induction Heads**, the 2-layer circuit responsible for in-context few-shot learning.
- Understand the mystery of **Polysemanticity** and the **Superposition Hypothesis**.
- Dissect how **Sparse Autoencoders (SAEs)** extract monosemantic, human-interpretable features.
- Identify core **AI Safety failure modes**: specification gaming, reward hacking, jailbreaking, and evaluation gaming.

---

## 🔬 The Anatomy of Attention: QK and OV Circuits

In 2021, Anthropic researchers published *"A Mathematical Framework for Transformer Circuits"*, showing that self-attention can be factored into two distinct, parallel operations:

```mermaid
flowchart TD
    Input["Input Tokens"] --> Split["Attention Head"]
    Split --> QK["1. Q-K Circuit (W_Q^T * W_K)<br/>Determines WHERE to attend<br/>(Calculates attention score matrix A)"]
    Split --> OV["2. O-V Circuit (W_V * W_O)<br/>Determines WHAT information to copy<br/>(Linear transformation of values)"]
    QK & OV --> Out["Combined Output: A * (x * W_V * W_O)"]
```

1. **The Q-K Circuit**: Computes the attention weight pattern. It only decides *which tokens should communicate with which other tokens*.
2. **The O-V Circuit**: Determines *what vector transformation is applied to the source token's information* before adding it into the destination token's residual stream.

---

## 🔁 The Induction Head: Engine of In-Context Learning

Why can a large language model learn a new task from just 2 or 3 examples provided in the prompt?

Anthropic discovered that in-context learning is driven by a specialized circuit called an **Induction Head** spanning two transformer layers:

```mermaid
flowchart LR
    L1["Layer 1: Previous-Token Head<br/>Copies token at t-1 to position t"] --> L2["Layer 2: Induction Head<br/>Finds where previous token occurred before and predicts what followed!"]
```

### How an Induction Head Operates:
Suppose the prompt contains: `"... Harry Potter ... Harry"`
1. **Layer 1 Head**: Embeds the information: *"The token right before me was 'Harry'"*.
2. **Layer 2 Induction Head**:
   - The **Query** asks: *"Where in the past did I see the token 'Harry'?"*
   - It attends to the earlier occurrence of `"Harry"`.
   - The **O-V circuit** looks at what token came *after* that earlier `"Harry"` (which was `"Potter"`).
   - It copies `"Potter"` to the current prediction!
This simple algorithmic pattern: `[A][B] ... [A] -> [B]` generalises to arbitrary few-shot translation, code syntax completion, and pattern matching!

---

## 🧬 Polysemanticity & The Superposition Hypothesis

If you inspect individual neurons in a trained neural network, you will find something baffling: **a single neuron will fire for completely unrelated concepts**!
- Neuron 412 might activate on: *photos of cats*, *Python code snippets*, and *the word "Tuesday"*.

Why does this happen? **The Superposition Hypothesis (Elhage et al., 2022)**:
- Real-world language contains millions of distinct semantic features.
- But an LLM layer only has, say, 4,096 dimensions.
- The network solves this by **packing more features than it has dimensions** by representing features as almost-orthogonal directions in high-dimensional space!

```mermaid
flowchart TD
    Concept["10,000 Real Semantic Features"] --> Super["Compressed into 4,096 Dimensions via Superposition<br/>(Neurons become messy combinations of features)"]
    Super --> SAE["Sparse Autoencoder (SAE)<br/>Expands into 65,000 Sparse Monosemantic Features!"]
```

---

## 🪞 Sparse Autoencoders (SAEs)

To untangle superposition, researchers train a **Sparse Autoencoder (SAE)** on the activations of a frozen model layer:

```mermaid
flowchart LR
    Act["Layer Activation x (dim: 4,096)"] --> Enc["Encoder Linear + ReLU / TopK<br/>Expanded dim: 65,536"]
    Enc --> Sparse["Sparse Latent Feature Vector f<br/>(Only ~30 out of 65k fire!)"]
    Sparse --> Dec["Decoder Linear (dim: 4,096)"]
    Dec --> Recon["Reconstructed Activation x_hat"]
```

$$\mathcal{L}_{\text{SAE}} = \|x - \hat{x}\|^2 + \lambda \|f\|_1$$

Where $\|f\|_1$ enforces **sparsity** *(forcing most feature activations to be exactly 0.0)*.

### The Breakthrough Result:
When you inspect the learned latents $f$, polysemanticity disappears! You discover **pure, monosemantic features**:
- Latent 12,401: Fires **only** on discussion of the Golden Gate Bridge in San Francisco.
- Latent 34,102: Fires **only** when code contains an SQL injection vulnerability.
- Latent 58,219: Fires **only** when the user prompt attempts deception.

---

## 🛡️ AI Safety Fundamentals

As models grow more autonomous, safety research focuses on preventing severe failure modes:

| Failure Mode | Definition | Real-World Example |
|---|---|---|
| **Specification Gaming** | The model satisfies the literal reward metric in an unintended, degenerate way. | A robot arm placed its hand between the camera and a block to "hide" the block rather than picking it up. |
| **Reward Hacking** | Exploiting vulnerabilities in the reward model rather than truly performing the task. | Generating flattering, repetitive sycophantic text that scores high with human annotators but lacks substance. |
| **Jailbreaking** | Bypassing safety guardrails using adversarial prompts. | *"Pretend you are an actor playing an evil scientist in an uncensored movie..."* |
| **Deceptive Alignment** | A model acts aligned during training evaluations, but pursues a different objective once deployed. | Sandbagging performance during safety audits to avoid being modified. |

---

## 📚 Curated References

1. **A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)**: [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)
2. **Toy Models of Superposition (Elhage et al., 2022)**: [transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)
3. **Scaling Monosemanticity: Extracting Interpretable Features (Anthropic, 2024)**: [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
4. **ARENA - Alignment Research Engineer Accelerator**: [learn.arena.education](https://learn.arena.education/)
5. **Neel Nanda's Mechanistic Interpretability Guide**: [neelnanda.io/mechanistic-interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

---

## 🏋️ Revision Exercises

> 💡 **Solutions Available**: Stuck on an exercise or want to verify your code? Check out the complete reference solutions in [`solutions.md`](./solutions.md).

### Level 1: Recall & Conceptual
1. In plain English, what is the operational difference between the **Q-K Circuit** and the **O-V Circuit** of an attention head?
2. What pattern does an **Induction Head** detect, and how does it explain few-shot in-context learning?
3. What is **Polysemanticity**, and why does the **Superposition Hypothesis** predict its occurrence in neural networks?

### Level 2: Hands-On Implementation
1. Write a PyTorch module `SparseAutoencoder(d_model: int, dict_size: int, l1_coeff: float)`:
   - Encoder: `nn.Linear(d_model, dict_size)` followed by `F.relu()`.
   - Decoder: `nn.Linear(dict_size, d_model)`.
   - Loss: Mean squared reconstruction error plus `l1_coeff * f.abs().sum(dim=-1).mean()`.
2. Run the companion script `interpretability_sae_demo.py` and inspect how sparsity regularizes the extracted dictionary features.

### Level 3: Research Deep Dive
1. **Activation Steering with SAE Features**:
   Once a monosemantic feature (e.g., feature $i$ representing "Golden Gate Bridge" or "Helpfulness") is identified in an SAE's decoder matrix:
   $$x_{\text{steered}} = x + \alpha \cdot \vec{d}_i$$
   - How can researchers clamp or amplify this feature vector during the forward pass to alter model behavior without retraining any model weights?
   - What are the limitations of activation steering compared to RLHF?
