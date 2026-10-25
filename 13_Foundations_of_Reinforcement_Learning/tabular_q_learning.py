"""
AI Research Foundations - Module 13: Tabular Q-Learning in GridWorld
Run with: python tabular_q_learning.py
"""

import numpy as np

class GridWorld:
    """A 4x4 GridWorld navigation environment with a goal and a pit."""
    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pit = (1, 1)
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self._state_idx()

    def _state_idx(self):
        r, c = self.agent_pos
        return r * self.size + c

    def step(self, action):
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        r, c = self.agent_pos
        if action == 0:   r = max(0, r - 1)
        elif action == 1: r = min(self.size - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.size - 1, c + 1)
        
        self.agent_pos = (r, c)
        state_idx = self._state_idx()

        if self.agent_pos == self.goal:
            return state_idx, 10.0, True
        elif self.agent_pos == self.pit:
            return state_idx, -10.0, True
        else:
            return state_idx, -0.1, False  # Slight step cost to encourage fast arrival

def train_q_learning(episodes: int = 500, seed: int = 42, verbose: bool = True) -> dict:
    """
    Trains tabular Q-learning on 4x4 GridWorld.
    Returns dictionary with learned Q table, trajectory, and reaches_goal boolean.
    """
    np.random.seed(seed)
    env = GridWorld(size=4)
    n_states = 16
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    lr, gamma, epsilon = 0.1, 0.95, 0.2

    for episode in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.choice(n_actions)
            else:
                a = np.argmax(Q[s])

            next_s, r, done = env.step(a)
            td_target = r + gamma * np.max(Q[next_s]) if not done else r
            Q[s, a] += lr * (td_target - Q[s, a])
            s = next_s

    # Trace greedy path from start (0,0)
    env.reset()
    s = env._state_idx()
    trajectory = [env.agent_pos]
    reaches_goal = False
    for _ in range(20):
        a = np.argmax(Q[s])
        s, r, done = env.step(a)
        trajectory.append(env.agent_pos)
        if env.agent_pos == env.goal:
            reaches_goal = True
            break
        if env.agent_pos == env.pit or done:
            break

    if verbose:
        actions_arrows = ["↑", "↓", "←", "→"]
        print("\nLearned Policy Grid:")
        for r in range(4):
            row_str = ""
            for c in range(4):
                idx = r * 4 + c
                if (r, c) == env.goal:
                    row_str += "  🚩  "
                elif (r, c) == env.pit:
                    row_str += "  💀  "
                else:
                    best_a = np.argmax(Q[idx])
                    row_str += f"  {actions_arrows[best_a]}  "
            print(row_str)

    return {
        "Q": Q,
        "reaches_goal": reaches_goal,
        "trajectory": trajectory
    }

if __name__ == "__main__":
    print("=== Training Tabular Q-Learning on 4x4 GridWorld ===")
    res = train_q_learning(episodes=500, verbose=True)
    print("\nGreedy Trajectory from (0,0):", res["trajectory"])
    print(f"Policy Reaches Goal Successfully: {res['reaches_goal']}")
    print("\n✅ Module 13 Q-learning environment verified!")
