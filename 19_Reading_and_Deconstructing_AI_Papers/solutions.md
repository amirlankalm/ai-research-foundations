# 📝 Worked Solutions — Module 19: Reading, Deconstructing & Critiquing AI Papers

These worked solutions provide reference answers and complete worked examples for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Keshav's Three-Pass Method
- **Pass 1: The Bird's-Eye Scan (5–10 minutes)**:
  - *Objective*: Decide whether to read further, file away, or discard.
  - *Actions*: Read Title, Abstract, and Introduction; read section and subsection headings; read Conclusions; glance at references.
  - *Output*: Answers to the 5 Cs (Category, Context, Correctness, Contributions, Clarity).
- **Pass 2: Grasp Content & Evidence (~1 hour)**:
  - *Objective*: Understand the main mechanics, evidence, and figures without getting bogged down in proofs.
  - *Actions*: Scrutinize all diagrams, figures, and benchmark tables. Note unfamiliar symbols, terminology, or references for later lookup.
- **Pass 3: Virtual Re-Implementation (2–5 hours)**:
  - *Objective*: Fully deconstruct the paper to the point of being able to reproduce or virtualize every step.
  - *Actions*: Mentally derive every formula from scratch, anticipate algorithmic choices before reading them, identify unstated assumptions, and write prototype code.

### 2. What an Ablation Study Is and Why It Matters
An **Ablation Study** systematically removes or isolates individual components of a proposed system to measure each part's actual contribution to overall performance.
- While a headline benchmark table shows the final combined system, it obscures *why* the model improved.
- Ablations reveal whether performance gains were driven by the novel mathematical architecture, or whether they were simply artifacts of a larger training corpus, longer training steps, or a trivial hyperparameter tweak.

### 3. Benchmark Data Contamination
**Benchmark contamination** occurs when test set evaluation questions (or near-duplicates) are accidentally included in the web-scraped pretraining or fine-tuning dataset.
Because large language models excel at memorization, an evaluation on contaminated benchmarks measures rote recall rather than genuine multi-step reasoning or mathematical capability, creating an illusion of reasoning prowess that collapses when tested on private, freshly constructed benchmarks.

---

## Level 2: Hands-On Implementation

### Reference Pass 1 Critique for DeepSeekMath (arXiv:2402.03300)

```markdown
# 5 C's Critique: DeepSeekMath (arXiv:2402.03300)

1. Category:
   Post-training reinforcement learning and pretraining data curation for mathematical reasoning in open LLMs.

2. Context:
   Builds directly on DeepSeek-Coder-Base (7B), traditional PPO (Schulman et al., 2017), and self-instruct mathematical corpora (e.g. GSM8K, MATH).

3. Correctness:
   Methodology is sound and validated against multiple public benchmarks (GSM8K, MATH, SAT, Olympiad). Uses deterministic Python and symbolic math rule-based verifiers to eliminate subjective human reward modeling bias.

4. Contributions:
   - DeepSeekMath-Base-7B: Pretrained on 120B math-specific tokens extracted via an iterative web pipeline from Common Crawl.
   - Group Relative Policy Optimization (GRPO): A critic-free RL algorithm that computes policy advantage relative to group response mean and std dev, saving ~50% GPU memory compared to PPO.
   - Proves that rule-based outcome verifiers combined with RL scale mathematical reasoning without reward models.

5. Clarity:
   Exceptionally clear and transparent. Section 3 details the GRPO objective with explicit math formulas, clipping parameters, and comparison tables against standard PPO.
```

---

## Level 3: Research Deep Dive

### 1. Evaluating Evaluation Metrics: Pass@1, Maj@K, and Pass@K
- **Greedy Decoding (Pass@1)**: The model produces a single deterministic sequence by selecting the token with maximum probability ($\text{temperature} = 0.0$) at every step. This represents typical production inference cost ($1\times$).
- **Maj@K (Majority Voting)**: The model samples $K$ independent stochastic reasoning traces (e.g. $K = 64$ with $\text{temperature} = 0.7$). The final answers are extracted, and the most common answer is selected by majority vote. This filters out stochastic calculation errors but requires $K\times$ inference compute.
- **Pass@K (Best of K)**: The model samples $K$ reasoning traces. If **at least one** of the $K$ traces arrives at the correct answer, the run is considered successful:
  $$\text{Pass}@K = \mathbb{E}\left[ 1 - \frac{\binom{N-C}{K}}{\binom{N}{K}} \right]$$
- **How Scores Are Manipulated**: An author claiming "90% on GSM8K" might quietly report **Pass@64** (where the model has 64 lottery tickets to stumble onto the correct answer) while comparing against competitor models evaluated strictly on **Pass@1** (greedy single-shot). Comparing Pass@64 against Pass@1 conflates search compute with raw model capability. Always check whether benchmark numbers are greedily evaluated or multi-sample consensus!
