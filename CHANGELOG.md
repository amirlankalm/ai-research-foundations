# Changelog (`CHANGELOG.md`)

All notable changes to the **AI Research Foundations** curriculum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-23

### Added
- Modular test suite split into [`tests/test_core.py`](tests/test_core.py) (zero PyTorch dependencies for lightweight installations) and [`tests/test_torch.py`](tests/test_torch.py) (with `pytest.importorskip("torch")` and finite gradient backprop checks).
- Worked solution guides ([`solutions.md`](01_Python_and_Research_Tooling/solutions.md)) for Level 1 and Level 2 exercises across Modules 01, 02, 03, 04, and 05.
- Google Colab interactive notebook pilot for Modules 01, 02, and 03 with "Open in Colab" badges.
- Fully worked tiny nanoGPT language modeling experiment in Module 12 with hand-rolled `CausalSelfAttention`, fixed random seed, theoretical baseline comparison, and empirical held-out validation metrics.
- Time estimates and "What 'Done' Looks Like" completion checklists across modules.
- Alphabetical AI research glossary ([`GLOSSARY.md`](GLOSSARY.md)) and beginner error diagnostic guide ([`STUCK.md`](STUCK.md)).
- Tiered dependency management ([`requirements-core.txt`](requirements-core.txt) vs. [`requirements-torch.txt`](requirements-torch.txt)).
- GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) testing matrix on Python 3.11 and 3.12.
- Enabled GitHub Discussions and published 5 curated starter issues labeled `good first issue`.

### Changed
- Reframed curriculum positioning honestly: "From Python to Frontier Reasoning Models" with clear prerequisite note (designed after 30-Days-Of-Python or CS50P).
- Renamed Capstone script to `grpo_loss_walkthrough.py` with honest walkthrough framing across docstrings, console outputs, and documentation.
- Replaced hardcoded test badge with real live GitHub Actions CI status badge.
- Reduced Paperleet mentions to a single balanced listing under Master Resources.
