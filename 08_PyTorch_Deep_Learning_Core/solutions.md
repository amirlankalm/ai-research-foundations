# 📝 Worked Solutions — Module 08: PyTorch Deep Learning Core

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Why `optimizer.zero_grad()` is Required
In PyTorch, gradients accumulate in `.grad` attributes by default across successive calls to `.backward()`. This design supports techniques like **gradient accumulation** across mini-batches when GPU memory is limited.
However, if you forget to zero out gradients at the start of each training iteration, the gradients from the new mini-batch are added to the gradients of the previous batches. The step will update weights in an erratic, exploded direction, causing training to fail or diverge. `optimizer.zero_grad(set_to_none=True)` is the recommended modern idiom, resetting `.grad = None` which reclaims memory.

### 2. Difference Between `model.train()` and `model.eval()`
- `model.train()` sets `self.training = True` recursively on all modules, enabling training-specific behaviors like random dropout masks and moving-average updates in batch normalization.
- `model.eval()` sets `self.training = False`, disabling stochastic behaviors to produce deterministic outputs.
- **Two layers affected**:
  1. `nn.Dropout`: Drops neurons with probability $p$ and scales remaining weights by $\frac{1}{1-p}$ during `train()`; passes inputs through as an identity operation during `eval()`.
  2. `nn.BatchNorm1d` / `nn.BatchNorm2d`: Computes mean and variance over the current mini-batch and updates running exponential statistics during `train()`; freezes updates and normalizes using precomputed running statistics during `eval()`.

### 3. Wrapping Evaluation in `with torch.no_grad():`
By default, PyTorch allocates a dynamic computational graph during the forward pass to track intermediate activations needed for backpropagation.
In evaluation/inference, backpropagation is never called. Wrapping code in `with torch.no_grad():` disables autograd graph construction, dramatically reducing GPU/CPU memory consumption and accelerating execution speed.

---

## Level 2: Hands-On Implementation

### Concentric Circles MLP with Checkpointing

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
import numpy as np

class CirclesDataset(Dataset):
    def __init__(self, n_samples: int = 1000, noise: float = 0.05, factor: float = 0.5):
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CirclesMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & Loaders
    dataset = CirclesDataset(n_samples=1200)
    train_data, val_data = torch.utils.data.random_split(dataset, [1000, 200])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # 2. Model, Criterion, Optimizer
    model = CirclesMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_data)

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                val_loss += criterion(preds, y_b).item() * len(X_b)
                correct += (preds.argmax(dim=-1) == y_b).sum().item()
        val_loss /= len(val_data)
        val_acc = correct / len(val_data)

        # Checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, "best_circles_model.pt")

        if epoch % 5 == 0 or epoch == 20:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}%")

if __name__ == "__main__":
    train_and_evaluate()
```

---

## Level 3: Research Deep Dive

### 1. Dynamic vs. Static Graphs & `torch.compile()`
- **Dynamic Graphs (Eager Mode)**: In PyTorch, every line of forward pass code builds a new directed acyclic graph on the fly. This enables native Python control flow (`if`, `for`, list comprehensions), making debugging trivial with standard tools (`pdb`, `print()`, conditional breakpoints). Variable-length sentences or recursive tree structures do not require complex static-graph padding or symbolic control constructs (`tf.while_loop`).
- **`torch.compile()` in PyTorch 2.0**:
  - Uses **TorchDynamo** to intercept Python bytecode at runtime and extract straight-line subgraphs ("FX graphs") of pure PyTorch operations, falling back gracefully to Python interpreter execution for non-PyTorch code.
  - Passes the captured FX graph to **TorchInductor**, an optimizing compiler that performs operator fusion (e.g., fusing consecutive elementwise operations, LayerNorm, and activations into single GPU Triton kernels).
  - This eliminates memory bandwidth round-trips to GPU global memory while maintaining full eager Python flexibility and ergonomics.
