# 📝 Research Paper Critique & Reproduction Template

Use this structured markdown sheet whenever you read an AI paper or log a reproduction on **[Paperleet](https://paperleet.com)**.

---

## 📄 1. Metadata
- **Paper Title**: 
- **Authors**: 
- **Institution / Lab**: 
- **ArXiv URL**: 
- **Publication Date**: 
- **Code Repository**: 

---

## 🦅 2. Pass 1: The 5 C's Scan (10-minute pass)
- **Category**: *(Architecture / Optimization / Post-Training / Alignment / Empirical Survey)*
- **Context**: *(What prior papers does this directly build upon?)*
- **Correctness**: *(Are the assumptions reasonable? Does the setup look sound?)*
- **Contributions**: *(What are the 2-3 main claims in bullet points?)*
  1. 
  2. 
  3. 
- **Clarity**: *(Is the paper written clearly or needlessly obscured?)*

---

## 🔍 3. Pass 2: Deconstructing the Core Algorithm (1-hour pass)
- **Problem Statement**: What exact bottleneck does this paper solve?
- **Key Equation / Algorithm**: *(Paste and annotate the core mathematical formula in plain English)*:
- **Baseline Models Tested Against**: 
- **Primary Benchmarks & Metrics**: *(e.g. GSM8K, HumanEval, MMLU)*:
- **Ablation Study Insights**: *(What was learned by turning off components?)*:

---

## 🔬 4. Pass 3: Critique & Reproduction Notes (Deep Pass)
- **Hidden Assumptions & Omissions**: *(Did they omit hardware requirements, batch sizes, or hyperparameter sensitivities?)*
- **Potential Red Flags**: *(Weak baselines? Data leakage? Cherry-picked examples?)*
- **Key Hyperparameters Needed for Reproduction**:
  - Learning Rate:
  - Batch Size:
  - Number of Steps / Epochs:
  - Optimizer & Weight Decay:
- **Reproduction Result**: *(Did your independent run match their reported numbers?)*
