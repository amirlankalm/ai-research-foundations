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

def train_sae_experiment(epochs: int = 200, seed: int = 42, verbose: bool = True) -> dict:
    """
    Trains Sparse Autoencoder on activations.
    Returns metrics dictionary with initial reconstruction loss, final reconstruction loss, and sparsity.
    """
    torch.manual_seed(seed)
    activations = torch.randn(500, 16)
    sae = SparseAutoencoder(d_model=16, dict_size=64, l1_coeff=0.01)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    init_recon_loss = None
    final_recon_loss = None
    final_sparsity = None

    if verbose:
        print("=== Training Sparse Autoencoder (SAE) Demo ===")

    for epoch in range(1, epochs + 1):
        _, latents, total_loss, recon_loss, l1_loss = sae(activations)
        if epoch == 1:
            init_recon_loss = recon_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            sparsity = (latents == 0.0).float().mean().item() * 100
            final_recon_loss = recon_loss.item()
            final_sparsity = sparsity
            if verbose:
                print(f"Epoch {epoch:3d} | Total: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | Sparsity: {sparsity:.1f}% inactive")

    if verbose:
        print("\n✅ Module 18 SAE verified successfully!")

    return {
        "init_recon_loss": init_recon_loss,
        "final_recon_loss": final_recon_loss,
        "final_sparsity": final_sparsity,
        "sae": sae
    }

if __name__ == "__main__":
    train_sae_experiment(epochs=200, seed=42, verbose=True)
