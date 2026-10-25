"""
AI Research Foundations - Module 08: Canonical PyTorch Training & Eval Pipeline
Run with: python pytorch_pipeline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ResearchMLP(nn.Module):
    """Clean reference PyTorch MLP with GELU and LayerNorm."""
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def run_training_pipeline(epochs: int = 5, seed: int = 42, device_name: str = None) -> dict:
    """
    Executes canonical PyTorch training & validation loop.
    Returns metrics dictionary with initial loss, final loss, and validation accuracy.
    """
    torch.manual_seed(seed)
    if device_name is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(device_name)

    # 1. Create synthetic classification data (non-trivial separable clusters)
    X = torch.randn(500, 10)
    y = ((X[:, 0] > 0).long() + (X[:, 1] > 0).long())  # 3 classes: 0, 1, 2
    train_ds = TensorDataset(X[:400], y[:400])
    val_ds = TensorDataset(X[400:], y[400:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Initialize Model, Optimizer, Loss
    model = ResearchMLP(in_features=10, hidden_dim=32, num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    # 3. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc*100:.1f}%")

    return {
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "val_acc": val_acc,
        "model": model
    }

if __name__ == "__main__":
    print("=== PyTorch Training Loop Sanity Check ===")
    res = run_training_pipeline(epochs=5)
    print("\n✅ Module 08 PyTorch pipeline successfully verified!")
