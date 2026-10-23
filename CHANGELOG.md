# Changelog (`CHANGELOG.md`)

All notable changes to the **AI Research Foundations** curriculum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - `quality-pass` branch

### Added
- Comprehensive repository audit report ([`AUDIT.md`](AUDIT.md)).
- Complete alphabetical AI research index ([`GLOSSARY.md`](GLOSSARY.md)) covering every defined term with direct chapter links.
- Beginner troubleshooting guide ([`STUCK.md`](STUCK.md)) diagnosing the top 15 PyTorch and numerical computation errors.
- Copyright and licensing notice ([`ASSETS_NOTICE.md`](ASSETS_NOTICE.md)) clarifying third-party artwork exclusions for `assets/banner.png`.
- Two-tier dependency specifications: [`requirements-core.txt`](requirements-core.txt) (NumPy-only modules) and [`requirements-torch.txt`](requirements-torch.txt) (Deep learning modules).
- Automated test suites in [`tests/`](tests/) running Level 2 implementation exercises via `pytest`.
- GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) testing CPU-safe scripts and tests.
- Release checklist ([`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)), issue drafts ([`ISSUE_DRAFTS.md`](ISSUE_DRAFTS.md)), and private launch notes draft ([`LAUNCH_NOTES.md`](LAUNCH_NOTES.md)).

### Changed
- Renamed Module 20 from *"Capstone Paper Reproduction"* to *"Capstone: GRPO Loss Walkthrough & Mathematical Foundations"* to accurately reflect simulated group loss mechanics vs. full 7B cluster pretraining.
- Updated main `README.md` to remove unverified claims, replacing "definitive" with "visual, intuition-first path from Python to your first paper reproduction".
- Replaced star-history graph and star-begging blocks with clean, neutral open-source support notes.
- Calibrated Paperleet framing to highlight it as an optional LeetCode-style tool for AI researchers.

---

## [0.1.0] - 2026-10-23

### Added
- Initial 20-module curriculum structure covering Python tooling, Linear Algebra, Calculus, Probability, Classical ML, Micrograd, PyTorch, Deep Learning dynamics, BPE tokenization, Transformers, nanoGPT, Reinforcement Learning, PPO, SFT/DPO, RLHF/GRPO, Reasoning models, Interpretability, and Paper analysis.
- 20 high-resolution Matplotlib diagram schematics in `assets/`.
- Runnable demo Python scripts across modules.
