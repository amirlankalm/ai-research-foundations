# 📝 Worked Solutions — Module 01: Python & Research Tooling

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises. Attempt the exercises on your own before consulting these solutions!

---

## Level 1: Recall & Conceptual

### 1. Difference between `python -m venv .venv` and global `pip install`
- **Global `pip install`**: Installs packages into the operating system's shared Python site-packages directory. If Project A needs `torch==2.0` and Project B needs `torch==2.4`, installing one overwrites the other, leading to dependency conflicts ("dependency hell") and unrepeatable code.
- **Virtual Environment (`python -m venv .venv`)**: Creates an isolated sandbox containing its own Python interpreter and `site-packages` directory. Each research project maintains its exact versions, recorded in `requirements.txt`. Collaborators can replicate the exact environment without affecting their machine.

### 2. What does broadcasting mean?
**Broadcasting** is NumPy's and PyTorch's mechanism that allows element-wise operations on arrays of different shapes by conceptually expanding singleton dimensions (dimension size of 1) without allocating redundant memory copies.

### 3. Danger of committing a `.pt` model checkpoint to Git
Git is a content-addressable version control system designed for text files. Every time you commit a changed 2GB `.pt` weight file, Git saves the entire 2GB blob in the hidden `.git/objects` directory forever. Even if you delete the file later, the repository size remains massive, making cloning extremely slow and exceeding GitHub's 100MB per-file limit (causing push rejection). Always track large weights using `.gitignore` or Git LFS / Hugging Face Model Hub.

---

## Level 2: Hands-On Implementation

### 1. `config_manager.py` Implementation

```python
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    model_name: str
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: int = 1000
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass fields to a standard dictionary."""
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        """Serializes configuration to formatted JSON on disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"✅ Configuration successfully saved to {filepath}")

# Quick sanity test
if __name__ == "__main__":
    cfg = ExperimentConfig(model_name="nano-gpt-v1", learning_rate=1e-3, batch_size=64)
    print("Generated Config:", cfg)
    cfg.save_json("test_config.json")
```

### 2. `batch_sampler` Generator Function

```python
from typing import List, Any, Generator

def batch_sampler(data_list: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Yields successive batches of size `batch_size` from `data_list`.
    Uses slicing inside a generator so only one batch is yielded at a time,
    avoiding allocating redundant lists of all batches in memory.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

# Quick sanity test
if __name__ == "__main__":
    dataset = list(range(25))
    for i, batch in enumerate(batch_sampler(dataset, batch_size=10)):
        print(f"Batch {i}: {batch}")
    # Output:
    # Batch 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Batch 1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # Batch 2: [20, 21, 22, 23, 24]
```
