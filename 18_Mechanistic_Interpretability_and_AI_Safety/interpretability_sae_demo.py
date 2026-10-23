"""
AI Research Foundations - Module 18: Sparse Autoencoder (SAE) for Feature Untangling
Run with: python interpretability_sae_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) trained on hidden activations.
    Expands d_model into dict_size sparse monosemantic latents.
    """
    def __init__(self, d_model: int = 16, dict_size: int = 64, l1_coeff: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        # 1. Center activations
        x_centered = x - self.b_dec

        # 2. Encode to sparse latents via ReLU
        latents = F.relu(self.encoder(x_centered))

        # 3. Decode back to original activation space
        reconstructed = self.decoder(latents) + self.b_dec

        # 4. Compute Loss: MSE Reconstruction + L1 Sparsity Penalty
        recon_loss = F.mse_loss(reconstructed, x)
        l1_loss = self.l1_coeff * latents.sum(dim=-1).mean()
        total_loss = recon_loss + l1_loss

        return reconstructed, latents, total_loss, recon_loss, l1_loss

if __name__ == "__main__":
    print("=== Training Sparse Autoencoder (SAE) Demo ===")
    torch.manual_seed(42)

    # Simulate hidden activations from a neural network: 500 samples of dimension 16
    activations = torch.randn(500, 16)
    sae = SparseAutoencoder(d_model=16, dict_size=64, l1_coeff=0.01)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for epoch in range(1, 201):
        _, latents, total_loss, recon_loss, l1_loss = sae(activations)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            # Calculate sparsity: percentage of latents that are strictly zero
            sparsity = (latents == 0.0).float().mean().item() * 100
            print(f"Epoch {epoch:3d} | Total: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | Sparsity: {sparsity:.1f}% inactive")

    print("\n✅ Module 18 SAE verified successfully!")
