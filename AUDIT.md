# Repository Audit Report (`AUDIT.md`)

**Date:** October 2026  
**Auditor:** Senior Engineer-Educator (Automated Quality Pass)  
**Branch:** `quality-pass`  
**Test Hardware:** Apple Silicon (macOS) & Forced CPU Fallback (`CUDA_VISIBLE_DEVICES=""`)  
**Python Environment:** Python 3.11 with `requirements.txt`  

---

## 1. 20-Module Comprehensive Audit Table

| Module # & Name | README Present | Demo Script Present | Demo Runs on CPU? (Exact Command & Output Summary) | Exercises (L1/L2/L3) | Solutions Present? | Diagram Correct? | Unsupported Claims & Identified Issues |
|---|---|---|---|---|---|---|---|
| **01 Python & Research Tooling** | Yes | `tooling_demo.py` | **YES** (`python3 01_.../tooling_demo.py`, 0.03s)<br>`=== Testing ExperimentConfig ===`<br>`All Module 01 sanity checks passed!` | Yes (3/2/2) | **NO** (Need `solutions/` & `tests/`) | Yes (`01_dev_ecosystem.png`) | None. Pure Python standard library demo. |
| **02 NumPy & Data** | Yes | `numpy_foundations.py` | **YES** (`python3 02_.../numpy_foundations.py`, 0.12s)<br>`Broadcasting Demo ... Softmax Verification ... Cosine Similarity` | Yes (2/2/1) | **NO** | Yes (`02_tensor_broadcasting.png`) | None. Vectorized NumPy without torch. |
| **03 Linear Algebra** | Yes | `linear_algebra_demo.py` | **YES** (`python3 03_.../linear_algebra_demo.py`, 0.12s)<br>`SVD Low-Rank Approx ... sqrt(d_k) variance scaling` | Yes (3/2/1) | **NO** | Yes (`03_linear_algebra_visuals.png`) | None. Pure NumPy SVD and variance simulations. |
| **04 Calculus & Optimization** | Yes | `optimization_demo.py` | **YES** (`python3 04_.../optimization_demo.py`, 0.06s)<br>`Optimizing Rosenbrock: SGD, Momentum, Adam` | Yes (3/2/1) | **NO** | Yes (`04_optimization_landscapes.png`) | None. Pure NumPy 2D optimization paths. |
| **05 Probability & Information** | Yes | `probability_info_demo.py` | **YES** (`python3 05_.../probability_info_demo.py`, 0.05s)<br>`Entropy Analysis ... Cross-Entropy ... KL Asymmetry` | Yes (3/2/1) | **NO** | Yes (`05_information_theory.png`) | None. Pure NumPy information theory calculations. |
| **06 Classical ML** | Yes | `titanic_baseline.py` | **YES** (`python3 06_.../titanic_baseline.py`, 0.06s)<br>`Logistic Regression from Scratch ... Acc: 97.5%` | Yes (3/2/1) | **NO** | Yes (`06_ml_bias_variance.png`) | README references scikit-learn Titanic, demo is synthetic 2D scratch. |
| **07 Autograd (Micrograd)** | Yes | `micrograd_engine.py` | **YES** (`python3 07_.../micrograd_engine.py`, 0.05s)<br>`Scalar Autograd check ... Toy MLP ReLU XOR Loss: 0.078` | Yes (3/3/1) | **NO** | Yes (`07_autograd_computational_graph.png`) | None. 100-line scalar autograd works as claimed. |
| **08 PyTorch DL Core** | Yes | `pytorch_pipeline.py` | **YES** (`python3 08_.../pytorch_pipeline.py`, 4.26s CPU)<br>`5 Epochs ... Train Loss: 0.94 ... Val Acc: 42.0%` | Yes (3/1/1) | **NO** | Yes (`08_pytorch_ecosystem.png`) | None. Standard PyTorch training/eval loop on synthetic data. |
| **09 DL Dynamics & Norms** | Yes | `architectures_dynamics.py` | **YES** (`python3 09_.../architectures_dynamics.py`, 0.53s CPU)<br>`RMSNorm unit check ... SwiGLU forward pass` | Yes (3/2/1) | **NO** | Yes (`09_activations_and_norms.png`) | None. Verifies RMSNorm and SwiGLU blocks in PyTorch. |
| **10 Tokenization & BPE** | Yes | `bpe_tokenizer.py` | **YES** (`python3 10_.../bpe_tokenizer.py`, 0.02s)<br>`Mini BPE Train ... 31 bytes -> 17 tokens -> exact decode` | Yes (3/2/1) | **NO** | Yes (`10_bpe_tokenization.png`) | None. Pure Python byte-pair encoding from scratch. |
| **11 Attention & Transformer** | Yes | `attention_mechanisms.py` | **YES** (`python3 11_.../attention_mechanisms.py`, 0.53s CPU)<br>`Unmasked Attention ... Causal Mask ... PyTorch parity 0.00e+00` | Yes (3/2/1) | **NO** | Yes (`11_transformer_attention.png`) | None. Exact numerical parity with PyTorch reference. |
| **12 Building GPT (nanoGPT)** | Yes | `nanogpt_model.py` | **YES** (`python3 12_.../nanogpt_model.py`, 1.14s CPU)<br>`100 steps on toy Shakespeare ... Loss 3.34 -> 2.52 ... Text Gen` | Yes (3/2/1) | **NO** | Yes (`12_gpt_decoder_stack.png`) | Needs real loss curve & sample generation artifact. |
| **13 RL Foundations** | Yes | `tabular_q_learning.py` | **YES** (`python3 13_.../tabular_q_learning.py`, 0.07s)<br>`500 Episodes GridWorld ... Learned Policy Grid Arrows` | Yes (3/1/1) | **NO** | Yes (`13_rl_mdp_bellman.png`) | None. Exact tabular Q-learning grid convergence. |
| **14 Deep RL: PPO** | Yes | `clean_ppo.py` | **YES** (`python3 14_.../clean_ppo.py`, 0.53s CPU)<br>`PPO Clipped Surrogate Loss: -0.5366 ... ActorCritic sample` | Yes (3/2/1) | **NO** | Yes (`14_ppo_clipping.png`) | Single-step surrogate check; does not train full Atari/Gym agent. |
| **15 Post-Training: DPO** | Yes | `dpo_loss.py` | **YES** (`python3 15_.../dpo_loss.py`, 0.52s CPU)<br>`Initial DPO Loss: 0.6931 -> Improved: 0.4393` | Yes (3/2/1) | **NO** | Yes (`15_dpo_alignment.png`) | Verifies DPO loss calculation on sequence logps; not full SFT model run. |
| **16 RLHF & GRPO** | Yes | `grpo_core.py` | **YES** (`python3 16_.../grpo_core.py`, 0.53s CPU)<br>`Raw Rewards [1,0,1,0] -> Advantages [+1,-1,+1,-1] -> Loss 0.0118` | Yes (3/2/1) | **NO** | Yes (`16_deepseek_grpo.png`) | Core math is accurate; simulated logps only. |
| **17 Reasoning Models** | Yes | `reasoning_search_demo.py` | **YES** (`python3 17_.../reasoning_search_demo.py`, 0.03s)<br>`Pass@1: 39.8% -> Best-of-31 Majority Voting: 82.1%` | Yes (3/2/1) | **NO** | Yes (`17_reasoning_models.png`) | Synthetic search simulation, not real LLM inference. |
| **18 Interpretability & SAE** | Yes | `interpretability_sae_demo.py` | **YES** (`python3 18_.../interpretability_sae_demo.py`, 1.00s CPU)<br>`200 Epochs SAE ... Loss: 0.0381 ... Sparsity: 98.4% inactive` | Yes (3/2/1) | **NO** | Yes (`18_interpretability_sae.png`) | Trains SAE on synthetic activations, not live transformer layer. |
| **19 Paper Reading** | Yes | `paper_critique_template.md` | **N/A** (Template only, no runnable script) | Yes (3/3/1) | **NO** | Yes (`19_three_pass_reading.png`) | Missing executable Python demo for citation graph or arXiv search. |
| **20 Capstone: GRPO** | Yes | `grpo_loss_walkthrough.py` | **YES** (`python3 20_.../grpo_loss_walkthrough.py`, 0.98s CPU)<br>`4 Candidates ... Rewards [1.1, 1.1, 0.1, 0.0] ... GRPO Step 1..5 Loss: 0.00 -> -0.189` | Yes (3/3/2) | **NO** | Yes (`20_capstone_grpo_paper.png`) | **HONEST WALKTHROUGH:** Renamed from "reproduction" to standalone loss & group advantage walkthrough. |

---

## 2. Mathematical Audit of DeepSeekMath (arXiv:2402.03300)

We conducted a mathematical verification comparing our GRPO code (`16_.../grpo_core.py` and `20_.../grpo_loss_walkthrough.py`) directly against **Section 3 of arXiv:2402.03300**:

### Paper Formulation:
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min(r_{i,t} \hat{A}_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta D_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$

1. **Advantage Normalization**:
   - Paper: $\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$
   - Code: `(rewards - mean) / (std + eps)` -> **Mathematically Exact Match.**
2. **Probability Ratio & Clipping**:
   - Paper: $r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ clipped between $1-\epsilon$ and $1+\epsilon$ with $\epsilon = 0.2$.
   - Code: `torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv` -> **Mathematically Exact Match.**
3. **KL Divergence Estimator**:
   - Paper (Equation 4): $D_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$
   - Code: `torch.exp(logprobs_ref - logprobs_policy) - (logprobs_ref - logprobs_policy) - 1.0` -> **Exact Match with Schulman's $k_3$ estimator used in the paper.**
4. **Where the Code Diverges from a "Full Reproduction"**:
   - The paper fine-tunes a **7-Billion parameter DeepSeekMath-Base** model over thousands of GSM8K and MATH problems across a cluster of 8×A100 GPUs.
   - Our module script runs the exact GRPO loss equation, advantage standardizer, and rule-based verifier on a simulated candidate group of 4 outputs on CPU.
   - **Verdict:** The mathematical formulation in the code is correct, but claiming "Full reproduction on GSM8K" is an overclaim. It must be renamed to **"GRPO Loss Walkthrough & Mathematical Foundations"** (Option A).

---

## 3. Feasibility Note for Option B (Small-Scale Model Training)

- **Model Candidate:** SmolLM-135M or Qwen2.5-0.5B.
- **VRAM / RAM Requirement:** Even a 0.5B model in FP16/BF16 requires ~1.2 GB for weights, plus ~2.5 GB for optimizer states (AdamW), plus memory for caching activations for $G=4$ outputs of length 512 tokens. Total memory: ~6-8 GB.
- **Inference & Sampling Time:** Generating $G=4$ completions of length 256 tokens per prompt on a CPU takes ~45-90 seconds per single step. Running 100 training steps on CPU would take 2.5 to 4 hours.
- **Recommendation:** Implement Option A first (honest renaming, crystal-clear loss walkthrough, and test suite). Option B should remain a modular optional experiment with clear hardware caveats.

---

## 4. Banner Licensing Audit

- **Current Banner:** Contains images of characters D and Capri from *Cyberpunk: Edgerunners 2* (Studio Trigger / CD Projekt Red).
- **Issue:** The repository is licensed under MIT, but the banner contains third-party proprietary artwork.
- **Action Required:** Create `ASSETS_NOTICE.md` stating plainly that `assets/banner.png` is excluded from the MIT license, that third parties appear to own the artwork, that this notice clarifies the exclusion without granting reuse rights, and that permission from copyright holders is uncertain. Add a disclaimer in the main `README.md`.

---

## 5. Identified Action Items (Priority Order)

1. **Capstone Overclaim:** Rename Module 20 folder and all references from "Full Reproduction" to "GRPO Loss Walkthrough & Mathematical Foundations".
2. **Banner Notice:** Add `ASSETS_NOTICE.md` and banner exclusion clause in `README.md` and `LICENSE`.
3. **Template Standardization:** Ensure every module has Title, Prerequisites, Time Estimate, Intuition, Jargon Box, Math with worked numbers, Code Walkthrough, Real Output, Exercises (L1/L2/L3), Solutions, and Quiz.
4. **Requirements Split:** Create `requirements-core.txt` (NumPy, Scipy, Pandas, Matplotlib) and `requirements-torch.txt`.
5. **Support Pages:** Add `GLOSSARY.md`, `STUCK.md`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md`, `LAUNCH_NOTES.md`, and `ISSUE_DRAFTS.md`.
6. **Tests & CI:** Add `tests/` with `pytest` for all runnable exercises, and `.github/workflows/ci.yml`.
