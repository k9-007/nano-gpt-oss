# 04 — Attention Mechanisms

## The Evolution: MHA → MQA → GQA

### Multi-Head Attention (MHA) — Original

- 64 Q heads, 64 K heads, 64 V heads
- All heads have unique parameters
- KV cache: **LARGEST** (stores all 64 K and V heads per layer)

### Multi-Query Attention (MQA)

- 64 Q heads, **1** K head, **1** V head
- All Q heads share ONE KV pair
- KV cache: SMALLEST, but quality degrades significantly

### Grouped Query Attention (GQA) — GPT-OSS

- 16 Q heads, 4 KV heads (ratio 4:1)
- Every 4 Q heads share 1 KV head
- **Best tradeoff:** 4× smaller KV cache, negligible quality loss

```
MHA:  Q₁K₁V₁, Q₂K₂V₂, ..., Q₆₄K₆₄V₆₄    (64 unique KV)
MQA:  Q₁K₁V₁, Q₂K₁V₁, ..., Q₆₄K₁V₁       (1 shared KV)
GQA:  Q₁K₁V₁, Q₂K₁V₁, Q₃K₁V₁, Q₄K₁V₁,   (4 groups × 4 Q per KV)
      Q₅K₂V₂, Q₆K₂V₂, Q₇K₂V₂, Q₈K₂V₂, ...
```

### Why GQA Works (The Deep Insight)

**Attention is asymmetric:**
- K, V = "what exists" (memory/knowledge)
- Q = "what I want" (intent/question)

You can share memory (K,V), but you **cannot share intent** (Q).

Different Q vectors asking the same memory = different answers:
```
Q_head0 = "where is the subject?" → attends to token 0
Q_head1 = "what is the grammar?" → attends to token 2
Both use the SAME K and V, get DIFFERENT outputs!
```

### KV Cache Memory Savings

```
MHA:  64 heads × 64 dims × seq_len × 2 (K+V) = huge
GQA:  4 heads × 64 dims × seq_len × 2 (K+V) = 16× smaller!
```

---

## Sliding Window Attention

### Why Limit the Window?

- Full causal attention over 131K tokens = O(n²) = 17.1 billion ops
- Sliding window of 128 = O(n × w) = 16.8 million ops → **~1000× reduction**

### How It Works

```
Full causal: token at position 500 sees ALL tokens 0-499
Sliding (w=128): token at 500 sees only tokens 372-499

Mask comparison:
Full:     [1, 1, 1, 1, 1, 1, 1, 1]  ← sees everything past
Sliding:  [0, 0, 0, 0, 1, 1, 1, 1]  ← only last w=4 tokens
```

### Alternating Pattern in GPT-OSS

```
Layer 0:  Sliding (local patterns)
Layer 1:  Full (global context)
Layer 2:  Sliding
Layer 3:  Full
...
Layer 23: Full

# Even layers → sliding; Odd layers → full
self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
```

**Why alternate?** Information propagates through layers. Layer 1's window covers tokens 1-128, layer 2 (full) can now "see" all of those enriched representations. Indirect global reach through stacking.

---

## Rotary Positional Encodings (RoPE)

### Why Not Just Add Position Vectors?

| | Additive (GPT-2) | Rotary (RoPE) |
|---|-----------------|---------------|
| Changes magnitude? | Yes (adds to vector) | **No** (rotation preserves length) |
| Relative position? | No (absolute only) | **Yes** (dot product depends on distance) |
| Extrapolation? | Fails beyond training length | Better (with YaRN) |
| Semantic pollution? | Contaminates meaning | Clean separation |

### How RoPE Works

1. Pair up dimensions: (x₁, x₂), (x₃, x₄), ..., (x₆₃, x₆₄)
2. Each pair gets a different frequency: θ₁, θ₂, ..., θ₃₂
3. Rotate each pair by angle = frequency × position

```
For position p, pair i:
  angle = θᵢ × p

  [x_new]   [cos(angle)  -sin(angle)] [x_old]
  [y_new] = [sin(angle)   cos(angle)] [y_old]
```

### Why Rotation Encodes Relative Position

When computing Q·K^T:
```
Q at position m (rotated by mθ)
K at position n (rotated by nθ)

Q·K^T depends on rotation difference = (m-n)θ

Same relative distance (m-n) → same positional contribution
Regardless of absolute positions m and n!
```

### Frequency Pairs

```
θᵢ = 1 / (base^(2i/d))     where base = 500,000 (with YaRN)

Pair 0 (i=0): θ = 1.0       → rotates FAST (changes every token)
Pair 16 (i=16): θ = 0.001   → rotates SLOWLY (changes over 1000s of tokens)
Pair 31 (i=31): θ = 0.00001 → barely rotates (ultra long-range signal)

Fast pairs → local position (adjacent tokens)
Slow pairs → global position (paragraph-level structure)
```

### YaRN Scaling (Context Extension)

Trained on 4096 tokens but want to use 131K at inference:

```
For high-frequency pairs (local): interpolate (compress more rotations)
For low-frequency pairs (global): extrapolate (continue naturally)
For middle: blend interpolation + extrapolation

This lets RoPE work beyond its training length!
```

---

## Attention Sinks

### What Are They?

A learnable scalar per attention head that acts as a "garbage collector" for noisy attention.

### The Problem They Solve

In deep layers, some tokens don't have meaningful things to attend to. Without sinks, attention is forced to assign probability mass SOMEWHERE — creating noisy/wrong patterns.

### How They Work

```
1. Normal attention scores: (seq × seq) matrix
2. Append ONE extra column (the sink): (seq × seq+1)
3. Softmax across ALL columns including sink
4. Drop the sink column

The sink "absorbs" probability mass that would otherwise
go to irrelevant tokens. Remaining scores are cleaner.
```

### Why Not Just Ignore Positions?

- Softmax must sum to 1.0 — it HAS to put weight somewhere
- Better to have an explicit "put junk here" column
- Only costs 1 parameter per head (16 total for the whole model)

---

## Shapes Summary

```
QKV projection: (4096, 1024) @ W(1024, 1536) = (4096, 1536)
  Q: (4096, 1024) → reshape → (4096, 4, 4, 64)
  K: (4096, 256) → reshape → (4096, 4, 64)
  V: (4096, 256) → reshape → (4096, 4, 64)

Attention scores: (4, 4, 4096, 4096)  ← kv_heads × q_per_group × seq × seq
+ Sink: (4, 4, 4096, 4097)
After drop: (4, 4, 4096, 4096)

Weighted sum: (4096, 4, 4, 64) → flatten → (4096, 1024)
Output proj: (4096, 1024) @ W(1024, 1024) = (4096, 1024)
```
