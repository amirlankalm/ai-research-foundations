# 📝 Worked Solutions — Module 17: Reasoning Models & Test-Time Compute

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Why Intermediate Tokens Overcome Fixed-Depth Compute Limits
A transformer with $L$ layers performs a fixed number of matrix multiplications and FLOPs per generated token. For inherently serial computational problems (like circuit evaluation, mathematical induction, or graph traversal), the required depth of logic scales with problem size $O(n)$, whereas single-step generation forces all computation into a static $O(L)$ depth.
Generating intermediate chain-of-thought tokens converts sequential logic into time: each intermediate reasoning token passes through all $L$ layers and appends to the KV-cache, providing subsequent tokens with an effectively unbounded dynamic computational budget before producing the final terminal answer.

### 2. ORM vs. PRM (Outcome vs. Process Reward Models)
- **Outcome Reward Model (ORM)**: Evaluates only the final completed answer at the very end of the trajectory, assigning a binary or scalar score (e.g., $+1$ if correct, $-1$ if wrong). While cheap to train via programmatic verifiers, ORMs suffer from severe credit assignment ambiguity—a reasoning trace might make a catastrophic arithmetic blunder in step 2 and accidentally stumble into the right final number by coincidence.
- **Process Reward Model (PRM)**: Evaluates each individual reasoning step along the chain, assigning intermediate step scores $r(s_t)$. This pinpoints the exact step where an error occurred, enabling tree search (MCTS, beam search) to prune invalid reasoning branches early.

### 3. The "Aha Moment" in DeepSeek-R1-Zero
In the DeepSeek-R1-Zero paper (arXiv:2501.12948), an untrained base model was aligned purely via reinforcement learning with rule-based math and coding verifiers without supervised fine-tuning (SFT).
During training, the model autonomously learned to reconsider its current derivation, backtrack, and output explicit natural phrases such as:
> *"Wait, let me double-check this step... wait, that formula is invalid because $x$ cannot be negative... let me recalculate."*
This spontaneous emergence of self-correction and reflection without human demonstration data is famously dubbed the "Aha moment."

---

## Level 2: Hands-On Implementation

### 1. Best-of-N Search with Step Verification and Majority Voting

```python
import random
from collections import Counter

def mock_generator(problem: str):
    """Generates candidate reasoning paths with step-by-step checks."""
    # Problem: "Solve 3x + 12 = 27"
    # Ground truth: x = 5
    templates = [
        # Correct path
        {"steps": ["3x = 27 - 12", "3x = 15", "x = 5"], "ans": 5, "valid": True},
        # Faulty arithmetic path
        {"steps": ["3x = 27 - 12", "3x = 16", "x = 5.33"], "ans": 5.33, "valid": False},
        # Alternate correct path
        {"steps": ["x + 4 = 9 (dividing by 3)", "x = 9 - 4", "x = 5"], "ans": 5, "valid": True},
        # Sign error path
        {"steps": ["3x = 27 + 12", "3x = 39", "x = 13"], "ans": 13, "valid": False},
    ]
    weights = [0.4, 0.25, 0.2, 0.15]
    return random.choices(templates, weights=weights)[0]

def step_verifier(candidate) -> bool:
    """Verifies that every intermediate reasoning step is mathematically sound."""
    return candidate["valid"]

def best_of_n_search(problem: str, N: int = 10):
    candidates = [mock_generator(problem) for _ in range(N)]
    
    # Filter candidates by step verifier (PRM check)
    verified = [c for c in candidates if step_verifier(c)]
    
    if not verified:
        # Fallback to majority of all candidates if all fail verification
        verified = candidates
        
    answers = [c["ans"] for c in verified]
    most_common_ans, count = Counter(answers).most_common(1)[0]
    
    print(f"Generated {N} trajectories | Verified Valid: {len(verified)}")
    print(f"Selected Majority Consensus: {most_common_ans} (supported by {count}/{len(verified)} verified paths)")
    return most_common_ans

# Verification
if __name__ == "__main__":
    random.seed(42)
    selected = best_of_n_search("3x + 12 = 27", N=10)
    assert selected == 5, f"Expected 5, got {selected}"
```

---

## Level 3: Research Deep Dive

### 1. Test-Time Compute & The Bitter Lesson
- **Inference Search as The Bitter Lesson**: Rich Sutton's *Bitter Lesson* posited that human-engineered domain heuristics are consistently surpassed by general computation applied to **learning** (scaling parameters and pretraining data) and **search** (scaling test-time exploration). Pre-training scaling laws (Chinchilla) represent the learning dimension; test-time compute scaling (Best-of-N, tree search, beam search) represents the search dimension.
- **Diminishing Returns from Correlated Biases**:
  When scaling test-time search, sampling $N$ candidate paths from the same base model assumes candidate errors are conditionally independent given the prompt. In practice, foundation models share systematic blind spots, training corpus hallucinations, and misleading pattern shortcuts. If 90% of model samples fall into the identical misleading cognitive trap, majority voting among $10,000$ samples simply reinforces the erroneous consensus. Search scaling saturates unless coupled with independent verifiers or diverse model ensembles.
