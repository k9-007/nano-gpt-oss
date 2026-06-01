# 03 — Normalization & Embeddings

## Token Embeddings

### Why Embeddings (Not One-Hot)?

With vocab = 201,088, a one-hot vector is 201K-dimensional and **sparse**. Embeddings compress each token into a dense 1024-d vector where similar words cluster together after training.

```
One-hot "cat":  [0, 0, 0, ..., 1, ..., 0, 0]  ← 201,088 dims, mostly zeros
Embedding "cat": [0.23, -0.15, 0.82, ...]       ← 1024 dims, all meaningful
```

### GPT-2 vs GPT-OSS: Position Handling

| | GPT-2 | GPT-OSS |
|---|-------|---------|
| Method | Token embed + Absolute pos embed (added) | Token embed only |
| Position info | Added directly to embedding | Injected later via RoPE |
| Problem | Pollutes semantic meaning with position data | Clean separation |

```python
# GPT-2: adds position to token embedding
tok_embeds = self.tok_emb(input_ids)
pos_embeds = self.pos_emb(torch.arange(seq_len))
x = tok_embeds + pos_embeds  # ← pollution!

# GPT-OSS: embedding only, no position here
x = self.embedding(x)
# Position added later via RoPE in attention (rotation, not addition)
```

---

## RMS Normalization

### Why Normalize At All?

Without normalization, activations grow or shrink across layers → **vanishing/exploding gradients**. Normalization keeps values in a stable range.

### Why RMSNorm Over LayerNorm?

| | LayerNorm | RMSNorm |
|---|-----------|---------|
| Formula | `(x - mean) / std × γ + β` | `x / RMS(x) × scale` |
| Steps | Center (subtract mean) + Scale | Scale only |
| Parameters | Scale (γ) + Bias (β) | Scale only |
| Speed | Baseline | ~15% faster |
| Quality | Same | Same or slightly better |

### The Math

```
LayerNorm:
  norm(x) = (x - mean(x)) / sqrt(var(x) + ε)
  output = norm(x) × γ + β

RMSNorm:
  RMS(x) = sqrt(mean(x²) + ε)
  output = (x / RMS(x)) × scale
```

### Why Skipping the Mean Works

- The mean subtraction in LayerNorm "centers" the distribution
- But deep networks learn to compensate for any bias anyway
- Removing it saves one pass over the data and one parameter (β)
- Empirically, models train just as well without it

### The Code

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float32)
        )  # learnable scale, initialized to 1

    def forward(self, x):
        t = x.float()
        # Key: mean of SQUARES (not variance)
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(x.dtype)

# rsqrt = 1/sqrt — avoids computing sqrt then dividing
# No mean subtraction, no bias — just scale by 1/RMS
```

### Pre-Norm vs Post-Norm

GPT-OSS uses **Pre-Norm** (normalize BEFORE the sublayer):

```
Pre-Norm (GPT-OSS):     x → Norm → Attention → + residual
Post-Norm (GPT-2):      x → Attention → + residual → Norm
```

**Why Pre-Norm?**
- More stable gradients during training
- Can train deeper networks without divergence
- The residual stream stays "clean" (unnormalized) — makes learning easier
- Post-Norm can cause gradient explosion in very deep networks

---

## Shapes Through This Section

```
Input IDs:        (seq_len,)           = (4096,)
After embedding:  (seq_len, hidden)    = (4096, 1024)
After RMSNorm:    (seq_len, hidden)    = (4096, 1024)  ← same!
RMSNorm scale:    (hidden,)            = (1024,)
```
