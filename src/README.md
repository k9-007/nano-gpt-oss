# Source Code

This directory contains the standalone Python modules for the GPT-OSS architecture.

These are the same classes used in the training and inference notebooks, extracted here for reuse and clarity.

| File | Purpose |
|------|---------|
| `model.py` | Full GPT-OSS architecture — RMSNorm, RoPE, GQA Attention, SwiGLU, MoE, GPToss |
| `tokenizer.py` | Harmony tokenizer (extends O200K BPE with special tokens) |
| `config.py` | ModelConfig dataclass (all architecture + training hyperparameters) |
| `__init__.py` | Exports all classes for `from src import GPToss, ModelConfig, ...` |

> **Note:** The notebooks (`notebooks/train.ipynb` and `notebooks/inference.ipynb`) are self-contained and embed these classes directly for portability. The `src/` files are provided for those who want to import and use the model programmatically.

## Quick Usage

```python
from src import GPToss, ModelConfig, HarmonyTokenizer

config = ModelConfig()
model = GPToss(config, device="cuda")
tokenizer = HarmonyTokenizer()

ids = torch.tensor(tokenizer.encode("Once upon a time"))
logits = model(ids.cuda())  # (seq_len, 201088)
```
