"""
AI Research Foundations - Module 01: Python & Research Tooling Demo
Run with: python tooling_demo.py
"""

from dataclasses import dataclass, asdict
import json
from typing import List, Tuple, Generator

@dataclass
class ExperimentConfig:
    """Configuration container for an AI training experiment."""
    model_name: str = "nano-transformer"
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: int = 1000
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"✅ Configuration saved to {filepath}")

def batch_generator(data: List[int], batch_size: int) -> Generator[List[int], None, None]:
    """Yields batches of items without allocating new full-length lists in memory."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == "__main__":
    print("=== Testing ExperimentConfig ===")
    cfg = ExperimentConfig(learning_rate=1e-3, batch_size=64)
    print("Config Object:", cfg)
    cfg.save_json("config_example.json")

    print("\n=== Testing Batch Generator ===")
    sample_data = list(range(100))
    for step, batch in enumerate(batch_generator(sample_data, batch_size=20)):
        print(f"Step {step}: Batch of size {len(batch)} -> {batch[:5]}...")

    print("\n✅ All Module 01 sanity checks passed!")
