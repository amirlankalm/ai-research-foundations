"""
AI Research Foundations - Module 12: Complete Miniature GPT (nanoGPT) in PyTorch
Run with: python nanogpt_model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=65, block_size=64, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1 = nn.LayerNorm(n_embd),
                attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True),
                ln2 = nn.LayerNorm(n_embd),
                mlp = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd)
                )
            ))
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Causal mask for attention
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        for b in self.blocks:
            norm_x = b.ln1(x)
            attn_out, _ = b.attn(norm_x, norm_x, norm_x, attn_mask=causal_mask, is_causal=True)
            x = x + attn_out
            x = x + b.mlp(b.ln2(x))

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=30, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

if __name__ == "__main__":
    print("=== Training Miniature GPT on Synthetic Toy Text ===")
    torch.manual_seed(42)

    # 1. Toy text dataset
    raw_text = "To be, or not to be, that is the question: Whether tis nobler in the mind to suffer."
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)

    model = MiniGPT(vocab_size=vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # 2. Mini training loop
    for step in range(101):
        # Pick random chunk
        ix = torch.randint(len(data) - 16, (4,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+16+1] for i in ix])

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(f"Step {step:3d} | Cross-Entropy Loss: {loss.item():.4f}")

    # 3. Generate text
    start_context = torch.tensor([[stoi["T"], stoi["o"]]], dtype=torch.long)
    generated_ids = model.generate(start_context, max_new_tokens=40, temperature=0.7)
    generated_text = "".join([itos[i] for i in generated_ids[0].tolist()])
    print("\nGenerated Sample:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    print("\n✅ Module 12 nanoGPT verified!")
