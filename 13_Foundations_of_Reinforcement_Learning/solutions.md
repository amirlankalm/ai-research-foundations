# 📝 Worked Solutions — Module 13: Foundations of Reinforcement Learning

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. The Markov Property
- **Definition**: A stochastic process satisfies the Markov Property if the conditional probability distribution of future states depends solely upon the present state and action, and is conditionally independent of all past states and actions:
  $$\mathbb{P}(S_{t+1} = s' \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, \dots) = \mathbb{P}(S_{t+1} = s' \mid S_t = s_t, A_t = a_t)$$
- **Strictly Markovian Example**: Chess on a standard board. The legal moves and win probabilities depend solely on the current positions of the pieces on the 64 squares (plus en passant/castling rights), not on which sequence of moves led to that board.
- **Non-Markovian Example**: Driving a car based on a single static photograph without velocity sensors. You cannot determine whether the car is accelerating, braking, or which direction it is turning without historical frame context.

### 2. Discount Factor Extreme Cases ($\gamma = 0$ vs. $\gamma = 0.999$)
- **$\gamma = 0$**: The agent is completely myopic ("short-sighted"). It optimizes solely for immediate one-step reward $R_{t+1}$ and assigns zero value to any future outcomes.
- **$\gamma = 0.999$**: The agent is hyper-farsighted. It treats rewards 500 steps into the future almost as valuably as rewards right now ($\gamma^{500} \approx 0.606$). The agent is willing to endure substantial short-term penalties if they lead to massive long-term delayed rewards.

### 3. What Temporal Difference (TD) Error Measures
The TD Error $\delta_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$ measures the discrepancy or "surprise" between:
1. The **TD Target** ($R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$): the actual observed immediate reward plus the discounted estimated value of the next state.
2. The **Prior Belief** ($Q(S_t, A_t)$): what the agent predicted the state-action value would be before taking the action.
When $\delta_t > 0$, the transition was better than expected; when $\delta_t < 0$, it was worse.

---

## Level 2: Hands-On Implementation

### 1. $4 \times 4$ GridWorld with Cliff and Tabular Q-Learning

```python
import numpy as np

# Actions: 0: Up, 1: Down, 2: Left, 3: Right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_SYMBOLS = ['↑', '↓', '←', '→']

class GridWorld:
    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.cliff = (1, 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action_idx):
        dr, dc = ACTIONS[action_idx]
        nr = max(0, min(self.size - 1, self.state[0] + dr))
        nc = max(0, min(self.size - 1, self.state[1] + dc))
        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state == self.cliff:
            return self.state, -10.0, True
        else:
            return self.state, -0.1, False  # Small step penalty to encourage shortest path

def train_q_learning():
    np.random.seed(42)
    env = GridWorld()
    # Q-table: (4, 4, 4)
    Q = np.zeros((4, 4, 4))
    lr = 0.1
    gamma = 0.95
    epsilon = 0.2

    for episode in range(800):
        s = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(4)
            else:
                a = np.argmax(Q[s[0], s[1]])

            ns, r, done = env.step(a)
            best_next = np.max(Q[ns[0], ns[1]]) if not done else 0.0
            td_target = r + gamma * best_next
            Q[s[0], s[1], a] += lr * (td_target - Q[s[0], s[1], a])
            s = ns

    # Display optimal policy
    print("=== Optimal 4x4 GridWorld Policy Map ===")
    for r in range(4):
        row_str = ""
        for c in range(4):
            if (r, c) == env.goal:
                row_str += " G  "
            elif (r, c) == env.cliff:
                row_str += " X  "
            else:
                best_act = np.argmax(Q[r, c])
                row_str += f" {ACTION_SYMBOLS[best_act]}  "
        print(row_str)

if __name__ == "__main__":
    train_q_learning()
```

---

## Level 3: Research Deep Dive

### 1. The Curse of Dimensionality & Deep Q-Networks (DQN)
- **State Space Explosion in Atari**:
  A single frame is $210 \times 160$ pixels with 3 color channels ($256$ intensity levels per channel).
  The number of raw frame states is:
  $$256^{210 \times 160 \times 3} = 256^{100,800} \approx 10^{242,700}$$
  This vastly exceeds the number of atoms in the observable universe ($\sim 10^{80}$). Storing a tabular Q-table is physically impossible.
- **DeepMind's DQN Innovations (Mnih et al., 2015)**:
  1. **Neural Function Approximation**: Uses a Convolutional Neural Network $Q(s, a; \theta)$ parameterized by weights $\theta$ to map high-dimensional raw pixel inputs directly to scalar Q-values for each discrete action.
  2. **Experience Replay Buffer**: Consecutive video frames are highly correlated, which destabilizes gradient descent. Storing transitions $(s, a, r, s', \text{done})$ in a replay buffer and sampling random mini-batches breaks temporal correlations and enables data reuse.
  3. **Target Network ($\theta^-$)**: In standard Q-learning with neural networks, updating $\theta$ simultaneously shifts both the prediction $Q(s, a; \theta)$ and the target $r + \gamma \max_{a'} Q(s', a'; \theta)$, causing training instability ("chasing a moving target"). DQN maintains a separate target network whose parameters $\theta^-$ are frozen and updated only periodically.
