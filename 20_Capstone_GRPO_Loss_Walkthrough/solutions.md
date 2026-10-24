# 📝 Worked Solutions — Module 20: Capstone Walkthrough (DeepSeekMath GRPO)

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. The Three Components of the GRPO Objective
From Section 3 of arXiv:2402.03300:
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min(r_{i,t} \hat{A}_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right) \right]$$
The three essential components are:
1. **Surrogate Policy Ratio ($r_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$)**: Measures the change in probability mass between current and old policy.
2. **PPO-style Clipped Advantage**: Clips $r_{i,t}$ within $[1-\epsilon, 1+\epsilon]$ when multiplied by normalized group advantage $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}$, preventing destabilizing large policy updates.
3. **KL Divergence Penalty ($\beta \mathbb{D}_{\text{KL}}$)**: Penalizes divergence from the reference policy $\pi_{\text{ref}}$, preserving base linguistic and reasoning capabilities.

### 2. Elimination of Value Function Overestimation Bias
In traditional Actor-Critic PPO, advantage estimation relies on a neural Critic network $V_\phi(s)$ estimating expected return:
$$\hat{A}_t = R_t - V_\phi(S_t)$$
Neural value networks frequently suffer from function approximation errors, out-of-distribution hallucinations, and systematic overestimation bias in high-dimensional state spaces.
GRPO replaces the neural Critic entirely with **empirical group-relative normalization**:
$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$
Because advantages are calculated directly from the empirical distribution of responses $\{o_1, \dots, o_G\}$ sampled from the exact same prompt $q$, there is no auxiliary value network to hallucinate or drift.

### 3. Role of the Format Bonus Reward
In reasoning alignment, the verifier must extract the final numeric answer unambiguously (e.g. `<answer>42</answer>`).
A format reward assigns bonus points if the model's generation complies with structural tags (`<think>...</think><answer>...</answer>`).
Without format bonuses, the model might produce correct intermediate thoughts but omit tags, leading to a false 0 reward from the automated parser. The format bonus acts as an educational curriculum scaffold, ensuring the policy reliably outputs parseable reasoning trajectories.

---

## Level 2: Hands-On Implementation

### 1 & 2. Modified Verifier with Negative Numbers, Decimals, and Regex

```python
import re

def parse_and_verify_solution(generation: str, ground_truth: float, tolerance: float = 1e-4) -> tuple[float, bool]:
    """
    Extracts answer from <answer> tags and supports integers, negative numbers, and decimals.
    Returns (reward, is_correct).
    """
    reward = 0.0
    
    # 1. Format reward: Did the model use <think> and <answer> tags?
    has_format = bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", generation, re.DOTALL))
    if has_format:
        reward += 0.2

    # 2. Extract numeric content inside <answer>...</answer>
    match = re.search(r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", generation)
    if not match:
        return reward, False

    try:
        extracted_num = float(match.group(1))
        # Float tolerance check
        if abs(extracted_num - ground_truth) < tolerance:
            reward += 1.0
            return reward, True
    except ValueError:
        pass

    return reward, False

# Verification
if __name__ == "__main__":
    correct_sample = "<think>Let x = 10 - 15.5 = -5.5</think><answer>-5.5</answer>"
    rew, ok = parse_and_verify_solution(correct_sample, -5.5)
    print(f"Sample Output: Reward={rew:.2f}, Correct={ok}")
    assert ok and rew == 1.2
```

### 3. Ablation: Disabling KL Penalty ($\beta = 0.0$)
When setting $\beta = 0.0$ in `grpo_loss_walkthrough.py`:
- Without KL regularization, the policy ratio $r_t = \pi_\theta / \pi_{old}$ drifts rapidly away from 1.0.
- For high-reward responses, the probabilities increase unchecked until logits saturate; for zero-reward responses, logits are driven towards negative infinity.
- In long sequences, the absence of $\beta$ causes catastrophic entropy collapse: the policy becomes rigid and overconfident, losing grammatical diversity and ability to explore alternative reasoning paths.

---

## Level 3: Research Deep Dive

### 1. Scaling of Group Size $G$
- **Variance of Group Mean ($\bar{r}$)**:
  By the Central Limit Theorem, the variance of the sample mean scales inversely with sample size:
  $$\text{Var}(\bar{r}) = \frac{\sigma^2}{G}$$
  - When $G = 2$, the advantage baseline is extremely noisy. If both sampled answers happen to be wrong ($r_1 = 0, r_2 = 0$), the standard deviation is 0 and no learning occurs.
  - As $G$ increases to $G = 16$ or $G = 64$, the sample mean $\bar{r}$ reliably approximates the true expected success rate $\mathbb{E}[R \mid q]$, producing highly stable advantage signals.
- **Computational Trade-Off: Group Size $G$ vs. Prompt Batch $B$**:
  Given a fixed rollout generation budget of $M = B \times G$ samples:
  - **Larger $G$ (e.g. $G=64, B=8$)**: Provides extremely low-variance advantage estimates and higher exploration diversity per question, but visits fewer distinct mathematical concepts per batch.
  - **Larger $B$ (e.g. $G=4, B=128$)**: Exposes the model to a wide variety of problem types in each optimization step, but advantages are noisy and subject to higher variance.
  - DeepSeek found that $G \in [8, 16]$ with larger prompt batches $B$ represents the sweet spot for mathematical training throughput on modern GPU clusters.
