"""
AI Research Foundations - Module 12: Complete Miniature GPT (nanoGPT) in PyTorch
Hand-rolled Causal Multi-Head Self-Attention with zero black-box attention layers.
Run with: python nanogpt_model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Hand-rolled Multi-Head Causal Self-Attention.
    Projects Q, K, V in parallel, splits across heads, applies causal masking,
    and projects the concatenated heads back to embedding dimension.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Causal mask buffer: lower triangular matrix (allows attending to past and present tokens)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim (n_embd)

        # 1. Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # 2. Reshape into multi-head format: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # 4. Multiply weights by V and project out
        out = weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class TransformerBlock(nn.Module):
    """A standard Pre-LN Transformer decoder block."""
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Minimal Autoregressive Decoder-Only Transformer (nanoGPT style)."""
    def __init__(self, vocab_size: int = 65, block_size: int = 32, n_embd: int = 64, n_head: int = 4, n_layer: int = 2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying: share token embedding and output classifier weights
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 30, temperature: float = 0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def run_worked_experiment(steps: int = 150, verbose: bool = True) -> dict:
    """
    Complete, reproducible tiny character-level language modeling experiment:
    - Fixed seed: 42
    - Dataset: Hamlet soliloquy split 80% train / 20% validation
    - Uniform baseline: -ln(1/vocab_size)
    - Full training loop tracking train loss and held-out validation loss
    """
    torch.manual_seed(42)

    import os
    if os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()[:50000] # Use first 50k chars for quick CPU training
        if verbose:
            print("📖 Loaded local corpus 'input.txt'!")
    else:
        raw_text = (
            "To be, or not to be, that is the question: "
            "Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, "
            "Or to take arms against a sea of troubles "
            "And by opposing end them."
        )
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)

    # 80/20 train/validation split
    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    block_size = 16
    batch_size = 4
    model = MiniGPT(vocab_size=vocab_size, block_size=block_size, n_embd=48, n_head=3, n_layer=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    # Theoretical random baseline loss = ln(vocab_size)
    baseline_loss = math.log(vocab_size)
    if verbose:
        print("=" * 60)
        print("🔬 MODULE 12: WORKED TINY LANGUAGE MODEL EXPERIMENT")
        print("=" * 60)
        print(f"Dataset Size: {len(data)} characters | Vocabulary: {vocab_size} unique chars")
        print(f"Train/Val Split: {len(train_data)} train chars / {len(val_data)} val chars")
        print(f"Theoretical Uniform Baseline Loss: {baseline_loss:.4f} (Perplexity: {vocab_size:.1f})")

    def get_batch(split: str):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x, y

    # Initial untrained evaluation
    with torch.no_grad():
        x_init, y_init = get_batch("val")
        _, init_val_loss = model(x_init, y_init)
    if verbose:
        print(f"Initial Step 0 Val Loss: {init_val_loss.item():.4f}")

    val_loss = init_val_loss
    # Training loop
    for step in range(1, steps + 1):
        x_batch, y_batch = get_batch("train")
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                x_val, y_val = get_batch("val")
                _, val_loss = model(x_val, y_val)
            if verbose:
                print(f"Step {step:3d} | Train Loss: {loss.item():.4f} | Held-out Val Loss: {val_loss.item():.4f}")

    # Final generation
    context = torch.tensor([[stoi["T"], stoi["o"]]], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=40, temperature=0.7)
    sample_text = "".join([itos[i] for i in generated[0].tolist()])
    if verbose:
        print("-" * 60)
        print("Generated Text (Prompt: 'To'):")
        print(sample_text)
        print("-" * 60)
        print("✅ Module 12 experiment completed successfully!")

    return {
        "baseline_loss": baseline_loss,
        "init_val_loss": init_val_loss.item(),
        "final_train_loss": loss.item(),
        "final_val_loss": val_loss.item(),
        "generated_sample": sample_text,
        "model": model
    }


if __name__ == "__main__":
    run_worked_experiment()
