# 📝 Worked Solutions — Module 10: Tokenization & Byte-Pair Encoding (BPE)

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Byte-Level BPE vs. Character-Level BPE
- **Character-Level BPE**: Starts from the set of Unicode characters present in the training set. Because Unicode contains over 150,000 characters (including emojis, non-Latin alphabets, and obscure symbols), character-level tokenizers either start with a massive initial vocabulary or inevitably encounter **Out-Of-Vocabulary (OOV)** `<unk>` tokens when faced with unseen foreign characters or emojis.
- **Byte-Level BPE**: Starts from the 256 fundamental bytes ($0x00$ through $0xFF$) of UTF-8 encoding. Any valid text, code, emoji, or arbitrary binary data can be represented as a sequence of bytes. Therefore, byte-level BPE guarantees **zero out-of-vocabulary tokens** forever while keeping the initial base vocabulary small (exactly 256 tokens).

### 2. String Reversal Failure on Token `"computer"`
When text is converted into tokens, the word `"computer"` is tokenized into a single atomic integer (e.g., token ID `15201`). The neural network never observes the individual constituent letters `'c', 'o', 'm', 'p', 'u', 't', 'e', 'r'` in its input sequence; it only receives the embedded vector corresponding to `15201`.
Asking the model to "reverse this word" requires it to reconstruct the spelling purely through implicit associative memory in its feedforward layers rather than direct positional attention over the individual letters. This explains why LLMs struggle with character counting, anagrams, and string reversal tasks.

### 3. Embedding Table vs. Standard Linear Layer
- Mathematically, an **Embedding Table** $\mathbf{E} \in \mathbb{R}^{V \times D}$ is equivalent to a linear layer without bias applied to a one-hot encoded vector $\mathbf{x} \in \mathbb{R}^V$:
  $$\mathbf{h} = \mathbf{x} \mathbf{E}$$
- Computationally, multiplying a $1 \times V$ one-hot vector (e.g. $V = 100,000$) by a $V \times D$ matrix is wasteful since all entries are 0 except one. PyTorch's `nn.Embedding` implements this as an $O(1)$ memory lookup: indexing row `token_id` directly from memory without matrix multiplications.

---

## Level 2: Hands-On Implementation

### 1. BPE `encode` and `decode` Functions

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_bpe(text: str, vocab_size: int = 270):
    tokens = list(text.encode("utf-8"))
    num_merges = vocab_size - 256
    merges = {}  # (p0, p1) -> new_id
    vocab = {i: bytes([i]) for i in range(256)}

    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        idx = 256 + i
        tokens = merge(tokens, best_pair, idx)
        merges[best_pair] = idx
        vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]

    return merges, vocab

def encode(text: str, merges: dict) -> list[int]:
    """Encodes arbitrary text using trained BPE merges."""
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Find the mergeable pair with lowest rank (first merged in training)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(token_ids: list[int], vocab: dict) -> str:
    """Decodes token IDs back into UTF-8 text."""
    byte_parts = [vocab[token] for token in token_ids]
    return b"".join(byte_parts).decode("utf-8", errors="replace")

# Verification
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog. The fox was quick."
    merges, vocab = train_bpe(sample_text, vocab_size=270)
    
    encoded = encode(sample_text, merges)
    decoded = decode(encoded, vocab)
    
    print(f"Original length: {len(sample_text.encode('utf-8'))} bytes")
    print(f"Encoded tokens:  {len(encoded)} tokens")
    print(f"Decoded matches: {decoded == sample_text}")
    assert decoded == sample_text
```

### 2. Tiktoken Inspection

```python
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    samples = ["apple", " strawberry", " 123456789"]

    print("\n=== Tiktoken cl100k_base Inspection ===")
    for s in samples:
        tokens = enc.encode(s)
        fragments = [enc.decode([t]) for t in tokens]
        print(f"Input: {repr(s):<16} | Tokens: {tokens} | Fragments: {fragments}")
except ImportError:
    print("tiktoken not installed (install via `pip install tiktoken`)")
```

---

## Level 3: Research Deep Dive

### 1. Token-Free Modeling & Quadratic Attention Bottleneck
- **Sequence Length Explosion**: Standard subword tokenizers compress natural text by approximately $3.5\times$ to $4.5\times$ (i.e. a 1,000-word paragraph is roughly 1,300 tokens, but spans $\sim 5,500$ UTF-8 bytes). Operating directly on raw bytes increases sequence length $N$ by roughly $4\times$.
- **Quadratic Attention Bottleneck**: In standard self-attention, time and memory scale as $O(N^2)$. A $4\times$ increase in sequence length $N$ causes a **$16\times$ increase** in attention memory and computation:
  $$\text{Memory}_{\text{attention}} \propto N^2 \cdot d_{\text{model}}$$
- This makes training context windows of 32k or 128k tokens prohibitively expensive on raw bytes without linear attention approximations, hierarchical downsampling, or state-space architectures (such as Mamba-Byte).
