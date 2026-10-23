"""
AI Research Foundations - Module 17: Test-Time Compute & Best-of-N Majority Voting Search
Run with: python reasoning_search_demo.py
"""

import random
from collections import Counter

def simulate_reasoning_attempt(problem_difficulty: float = 0.6) -> tuple[int, bool, str]:
    """
    Simulates a language model sampling a reasoning trajectory.
    Returns: (predicted_answer, step_by_step_validity, reasoning_trace)
    """
    correct_answer = 42
    # Chance of making an arithmetic slip during the chain
    if random.random() > problem_difficulty:
        trace = "<think> Step 1: Let x = 10. Step 2: 4*x + 2 = 42. </think> 42"
        return 42, True, trace
    else:
        hallucinated_answer = random.choice([38, 40, 44, 46])
        trace = f"<think> Step 1: Let x = 10. Step 2: calculation error... </think> {hallucinated_answer}"
        return hallucinated_answer, False, trace

def best_of_n_majority_vote(n_samples: int = 16) -> int:
    """Samples N reasoning paths and takes the consensus majority vote."""
    candidates = []
    for _ in range(n_samples):
        ans, is_valid, _ = simulate_reasoning_attempt()
        candidates.append(ans)
    
    votes = Counter(candidates)
    winner, count = votes.most_common(1)[0]
    return winner

if __name__ == "__main__":
    print("=== Test-Time Compute Scaling Simulation ===")
    random.seed(42)

    # 1. Single sample accuracy (pass@1 without test-time search)
    single_trials = [simulate_reasoning_attempt()[0] == 42 for _ in range(1000)]
    acc_single = sum(single_trials) / len(single_trials)
    print(f"Pass@1 Accuracy (Zero test-time compute): {acc_single * 100:.1f}%")

    # 2. Scaling test-time candidates (Best-of-N majority voting)
    for N in [3, 7, 15, 31]:
        search_trials = [best_of_n_majority_vote(n_samples=N) == 42 for _ in range(1000)]
        acc_search = sum(search_trials) / len(search_trials)
        print(f"Pass@Best-of-{N:2d} Majority Voting Accuracy: {acc_search * 100:.1f}%")

    print("\n✅ Module 17 reasoning search simulation complete!")
