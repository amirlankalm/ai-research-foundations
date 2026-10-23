# Contributing to AI Research Foundations

Thank you for your interest in contributing to **AI Research Foundations**! This project aims to be the most accessible, intuitive, and mathematically rigorous open-source on-ramp to modern AI research.

We welcome contributions of all kinds: fixing typos, improving visual diagrams, suggesting clearer physical analogies, adding new paper reproduction guides, or providing exercise solutions.

---

## 🧭 Ways to Contribute

1. **Clarifying Explanations & Analogies**: If an explanation felt too abstract or contained undefined jargon, submit an improvement. Our motto is: *Intuition first, plain English definitions for every term, zero math shock*.
2. **Fixing Typos & Code Bugs**: Found a bug in a PyTorch training loop or a typo in an equation? Open a Pull Request!
3. **Adding Exercise Solutions**: Add alternative solutions or extended experiments in the corresponding module directory.
4. **Reproductions & Paper Discussions**: Share your reproduction logs and findings on [Paperleet](https://paperleet.com) and link them in Module 19 / 20.

---

## 🛠️ Pull Request Workflow

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/ai-research-foundations.git
   cd ai-research-foundations
   ```
3. **Create a topic branch**:
   ```bash
   git checkout -b fix/module-04-gradient-descent
   # or
   git checkout -b feat/module-12-kv-cache-diagram
   ```
4. **Test your code**: Ensure all companion scripts execute cleanly without errors:
   ```bash
   python3 20_Capstone_Paper_Reproduction_DeepSeekMath_GRPO/reproduce_deepseek_grpo.py
   ```
5. **Commit your changes** with descriptive commit messages following the Conventional Commits specification:
   - `docs: fix typo in Module 03 dot product section`
   - `feat(day-11): add interactive KV-cache diagram`
   - `fix(day-08): update DataLoader worker initialization`
6. **Push to your fork** and open a Pull Request against the `main` branch.

---

## 📐 Style & Pedagogy Guidelines

When writing or modifying markdown chapters:
- **Tone**: Friendly, inspiring, educational, and technically precise.
- **Jargon Rule**: Always define new technical terms in parentheses or dedicated callouts on first mention, e.g.: *(gradient: a vector collecting all partial derivatives, pointing along the path of steepest increase)*.
- **Visuals**: Use Mermaid diagrams, ASCII layout charts, or Matplotlib figures for architecture flows.
- **Code**: Provide minimal, runnable, self-contained PyTorch or NumPy scripts that require no heavy external proprietary APIs.

---

## 📜 Code of Conduct

All contributors and community members are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please be kind, constructive, and welcoming to learners of all backgrounds!
