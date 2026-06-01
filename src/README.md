# Source Code

This directory contains the standalone Python modules for the GPT-OSS architecture.

These are the same classes used in the training and inference notebooks, extracted here for reuse and clarity.

| File | Purpose |
|------|---------|
| `model.py` | Full GPT-OSS architecture (RMSNorm, Attention, MoE, etc.) |
| `tokenizer.py` | Harmony tokenizer (extends O200K BPE) |
| `config.py` | Model configuration dataclass |

> **Note:** The notebooks (`notebooks/train.ipynb` and `notebooks/inference.ipynb`) are self-contained and embed these classes directly for portability. The `src/` files are provided for those who want to import and use the model programmatically.
