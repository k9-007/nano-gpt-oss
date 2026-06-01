# 06 — Training Pipeline & Inference

## Pre-Training Pipeline

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # peak learning rate
    betas=(0.9, 0.95), # momentum decay
    weight_decay=0.1,  # L2 regularization (decoupled)
    eps=1e-8
)
```

**Why AdamW over Adam?**
- Adam couples weight decay with gradient updates (wrong behavior)
- AdamW decouples them — applies weight decay directly to weights
- Results in better generalization (less overfitting)

### Learning Rate Schedule: Cosine Annealing with Warmup

```
LR
│
│    peak
│   ╱────╲
│  ╱      ╲
│ ╱        ╲  cosine decay
│╱          ╲───────────────
│  warmup    ╲            min_lr
└──────────────────────────────── steps

Phase 1 (warmup): linear ramp from 0 → peak LR over 2000 steps
Phase 2 (decay):  cosine decay from peak → min_lr
```

**Why warmup?**
- At start, gradients are random and large
- High LR + random gradients = unstable updates
- Warmup lets the model find a "good region" before taking big steps

**Why cosine decay?**
- Sharp start → big exploration early (escape bad local minima)
- Gentle end → fine refinement (don't overshoot good solutions)
- Smoother than step decay → no sudden training disruptions

```python
def get_lr(step, warmup_steps=2000, max_steps=100000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps  # linear warmup
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

### Gradient Accumulation

```
Want effective batch_size = 64, but GPU fits only batch_size = 8

Solution: accumulate gradients over 64/8 = 8 mini-batches
           then update weights once

for i in range(grad_accum_steps):
    loss = model(batch[i]) / grad_accum_steps  # scale loss
    loss.backward()  # gradients accumulate

optimizer.step()  # one weight update with all gradients
optimizer.zero_grad()
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- Prevents exploding gradients (especially with bfloat16)
- If gradient norm > 1.0, scale all gradients down proportionally
- Does NOT change direction, only magnitude

### Mixed Precision (bfloat16)

```python
model = model.to(dtype=torch.bfloat16)
```

- 16-bit floating point: 2× less memory, ~2× faster matmuls
- bfloat16 has same exponent range as float32 (unlike float16)
- No loss scaling needed (unlike float16 which can underflow)

---

## Training Configuration (H100)

```python
BATCH_SIZE = 64         # sequences per step
CONTEXT_LEN = 4096      # tokens per sequence
GRAD_ACCUM = 8          # effective batch = 64 × 8 = 512 sequences
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
MAX_STEPS = 100000
GRAD_CLIP = 1.0
```

---

## Inference & Sampling

### Auto-regressive Generation

```python
def generate(prompt_ids, max_new_tokens=100):
    for _ in range(max_new_tokens):
        logits = model(prompt_ids)       # (seq, vocab)
        next_logits = logits[-1]         # last position only
        next_token = sample(next_logits) # sampling strategy
        prompt_ids = concat(prompt_ids, next_token)
    return prompt_ids
```

### Temperature

Controls randomness:

```
temperature = 1.0  → normal distribution
temperature = 0.5  → sharper (more deterministic)
temperature = 2.0  → flatter (more random)
temperature → 0    → argmax (greedy, always picks highest)

logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

### Top-K Sampling

Only consider the K most probable tokens:

```
K = 50: only top 50 tokens are candidates
       all others get probability = 0

Prevents sampling very unlikely tokens (reduces "hallucination")
```

### Top-P (Nucleus) Sampling

Dynamic cutoff based on cumulative probability:

```
p = 0.9: include tokens until their cumulative probability ≥ 90%
         remaining tokens get probability = 0

Adapts to confidence:
  High confidence: few tokens needed to reach 90% → focused
  Low confidence: many tokens needed → diverse
```

### Combining Strategies

```python
def sample(logits, temperature=0.7, top_k=50, top_p=0.9):
    logits = logits / temperature
    
    # Top-K: keep only top 50
    top_k_logits, top_k_indices = logits.topk(top_k)
    
    # Top-P: within top-K, keep until cumulative prob ≥ 0.9
    probs = softmax(top_k_logits)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative - probs >= top_p
    probs[mask] = 0
    probs = probs / probs.sum()  # renormalize
    
    # Sample from filtered distribution
    idx = torch.multinomial(probs, 1)
    return top_k_indices[idx]
```

---

## Training Loop Structure

```python
for step in range(max_steps):
    # 1. Get learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 2. Gradient accumulation loop
    optimizer.zero_grad()
    total_loss = 0
    for micro_step in range(grad_accum_steps):
        input_ids, targets = get_batch()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        total_loss += loss.item()
    
    # 3. Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 4. Update weights
    optimizer.step()
    
    # 5. Log metrics
    if step % 100 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}, LR: {lr:.6f}")
    
    # 6. Save checkpoint
    if step % 5000 == 0:
        save_checkpoint(model, optimizer, step)
```
