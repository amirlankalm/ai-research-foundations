"""
AI Research Foundations - Module 10: Byte-Pair Encoding (BPE) Tokenizer From Scratch
Run with: python bpe_tokenizer.py
"""

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class MiniBPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int = 270):
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            tokens = merge(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            # Find the earliest learned merge pair
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        part_bytes = b"".join(self.vocab[idx] for idx in ids)
        return part_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    print("=== Training Mini BPE Tokenizer ===")
    sample_corpus = (
        "Transformer models process text through tokenization. "
        "Tokenization maps characters into integer IDs. "
        "Language models predict the next token given previous tokens."
    )
    tokenizer = MiniBPETokenizer()
    tokenizer.train(sample_corpus, vocab_size=275)

    test_sentence = "Language models predict tokens."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)

    print("Original Text:     ", test_sentence)
    print("Encoded Token IDs: ", encoded)
    print("Token Count:        Original bytes =", len(test_sentence.encode("utf-8")), "| BPE tokens =", len(encoded))
    print("Decoded Text:      ", decoded)
    assert decoded == test_sentence, "Decoding must exactly match original string!"
    print("\n✅ Module 10 BPE Tokenizer verified!")
