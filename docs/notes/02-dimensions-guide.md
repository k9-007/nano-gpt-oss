# 02 — Dimensions Guide: Every Matrix Shape Explained

## Tiny Model Config (For Tracing)

```
vocab_size     = 6    # (real: 201,088)
hidden_size    = 4    # (real: 1024)
seq_len        = 3    # (real: 4096)
num_attn_heads = 2    # (real: 16)
num_kv_heads   = 1    # (real: 4) — GQA ratio 2:1
head_dim       = 2    # (real: 64)

# Verify: num_attn_heads × head_dim = 2 × 2 = 4 = hidden_size ✓
# Q needs: 2 × 2 = 4 dims | K needs: 1 × 2 = 2 | V needs: 1 × 2 = 2
# Total QKV = 4 + 2 + 2 = 8
```

---

## Step 1: Embedding Lookup

```
Input: token IDs = [3, 1, 5]     shape: (3,)

Embedding table: shape (6, 4) — 6 vocab rows, 4 dims each

         dim0   dim1   dim2   dim3
token 0: [ 0.2   0.5  -0.1   0.3 ]
token 1: [ 0.8  -0.3   0.6   0.1 ]  ← ID=1
token 2: [ 0.4   0.7   0.2  -0.5 ]
token 3: [ 0.1   0.9  -0.4   0.7 ]  ← ID=3
token 4: [-0.2   0.1   0.8   0.4 ]
token 5: [-0.5   0.3   0.4   0.9 ]  ← ID=5

Output (pick rows 3, 1, 5):
x = [ 0.1   0.9  -0.4   0.7 ]   ← token 3
    [ 0.8  -0.3   0.6   0.1 ]   ← token 1
    [-0.5   0.3   0.4   0.9 ]   ← token 5

x shape: (3, 4) = (seq_len, hidden_size)
```

---

## Step 2: RMSNorm

```
Formula: x_norm = x / RMS(x) × scale
where RMS(x) = sqrt(mean(x²))

Row 0: [0.1, 0.9, -0.4, 0.7]

1. Square:  [0.01, 0.81, 0.16, 0.49]
2. Mean:    (0.01 + 0.81 + 0.16 + 0.49) / 4 = 0.3675
3. RMS:     sqrt(0.3675) = 0.606
4. Divide:  [0.1, 0.9, -0.4, 0.7] / 0.606 = [0.165, 1.485, -0.660, 1.155]
5. × scale: (assume scale = [1,1,1,1]) → [0.165, 1.485, -0.660, 1.155]

Input: (3, 4) → Output: (3, 4)  — shape NEVER changes
```

---

## Step 3: QKV Projection (Matrix Multiply)

```
W_qkv shape: (hidden_size, qkv_dim) = (4, 8)

W_qkv =  [ 0.3  -0.2   0.5   0.1  |  0.4  -0.1  |  0.2   0.6 ]
          [ 0.1   0.7  -0.3   0.4  |  0.2   0.5  | -0.3   0.1 ]
          [-0.4   0.2   0.6  -0.1  |  0.3   0.2  |  0.7  -0.2 ]
          [ 0.5  -0.1   0.2   0.8  | -0.2   0.4  |  0.1   0.5 ]
           ←──── Q (4 cols) ──────→  ←─ K (2) ─→  ←─ V (2) ─→

x_normed @ W_qkv: (3, 4) @ (4, 8) = (3, 8)

Row 0 computation (x_norm = [0.165, 1.485, -0.660, 1.155]):
  Col 0: 0.165×0.3 + 1.485×0.1 + (-0.660)×(-0.4) + 1.155×0.5
       = 0.050 + 0.149 + 0.264 + 0.578 = 1.040

  Col 1: 0.165×(-0.2) + 1.485×0.7 + (-0.660)×0.2 + 1.155×(-0.1)
       = -0.033 + 1.040 - 0.132 - 0.116 = 0.759

Result:
qkv = [ 1.04   0.76  -0.31   0.82  |  0.45  0.63  |  0.28  0.91 ]  ← token 0
      [ 0.55  -0.18   0.93   0.41  | -0.22  0.37  |  0.74 -0.15 ]  ← token 1
      [-0.27   0.64   0.47  -0.56  |  0.81 -0.44  | -0.32  0.58 ]  ← token 2
       ←──── Q (cols 0-3) ────────→ ←── K (4-5)─→ ←── V (6-7)──→
```

---

## Step 4: Split Q, K, V + Reshape for GQA

```
Q = qkv[:, 0:4]  → shape (3, 4)
K = qkv[:, 4:6]  → shape (3, 2)
V = qkv[:, 6:8]  → shape (3, 2)

GQA Reshape — reveal head structure:
Q: (3, 4) → (3, 1, 2, 2) = (seq, kv_groups, q_per_group, head_dim)

Token 0: Q flat = [1.04, 0.76, -0.31, 0.82]
  → head 0 = [1.04, 0.76]     ← "syntax query"
    head 1 = [-0.31, 0.82]    ← "semantic query"

K: (3, 2) → (3, 1, 2) = (seq, kv_heads, head_dim)
V: (3, 2) → (3, 1, 2) — same structure

Both Q heads SHARE the same K and V (that's GQA!)
```

---

## Step 5: Attention Scores (Q · K^T)

```
K vectors (shared by both heads):
  K[0] = [0.45, 0.63]
  K[1] = [-0.22, 0.37]
  K[2] = [0.81, -0.44]

HEAD 0 — Q vectors: [1.04, 0.76], [0.55, -0.18], [-0.27, 0.64]

  Q[0]·K[0] = 1.04×0.45 + 0.76×0.63 = 0.468 + 0.479 = 0.947
  Q[0]·K[1] = 1.04×(-0.22) + 0.76×0.37 = -0.229 + 0.281 = 0.053
  Q[0]·K[2] = 1.04×0.81 + 0.76×(-0.44) = 0.842 - 0.334 = 0.508

  Q[1]·K[0] = 0.55×0.45 + (-0.18)×0.63 = 0.248 - 0.113 = 0.134
  Q[1]·K[1] = 0.55×(-0.22) + (-0.18)×0.37 = -0.121 - 0.067 = -0.187
  Q[1]·K[2] = 0.55×0.81 + (-0.18)×(-0.44) = 0.446 + 0.079 = 0.525

  Q[2]·K[0] = (-0.27)×0.45 + 0.64×0.63 = -0.122 + 0.403 = 0.282
  Q[2]·K[1] = (-0.27)×(-0.22) + 0.64×0.37 = 0.059 + 0.237 = 0.296
  Q[2]·K[2] = (-0.27)×0.81 + 0.64×(-0.44) = -0.219 - 0.282 = -0.500

HEAD 0 score matrix:
           K[0]    K[1]    K[2]
  Q[0]: [ 0.947   0.053   0.508 ]
  Q[1]: [ 0.134  -0.187   0.525 ]
  Q[2]: [ 0.282   0.296  -0.500 ]

HEAD 1 — Different Q, SAME K → different scores (that's why GQA works!)
  Q[0]·K[0] = (-0.31)×0.45 + 0.82×0.63 = 0.377
  Q[0]·K[1] = (-0.31)×(-0.22) + 0.82×0.37 = 0.372
  Q[0]·K[2] = (-0.31)×0.81 + 0.82×(-0.44) = -0.612
  ...

Score shape: (1, 2, 3, 3) = (kv_heads, q_per_group, seq, seq)
```

---

## Step 6: Scale + Causal Mask

```
Scale: × 1/√head_dim = 1/√2 = 0.707

HEAD 0 scaled:
           K[0]    K[1]    K[2]
  Q[0]: [ 0.670   0.037   0.359 ]
  Q[1]: [ 0.095  -0.132   0.371 ]
  Q[2]: [ 0.199   0.209  -0.354 ]

Causal mask (token i cannot see j > i):
  [  0     -∞     -∞  ]
  [  0      0     -∞  ]
  [  0      0      0  ]

After mask:
           K[0]    K[1]    K[2]
  Q[0]: [ 0.670    -∞      -∞  ]   ← only sees self
  Q[1]: [ 0.095  -0.132    -∞  ]   ← sees 0, 1
  Q[2]: [ 0.199   0.209  -0.354]   ← sees all
```

---

## Step 7: Attention Sinks + Softmax

```
Sinks: learnable scalars, one per head
  head 0 sink = 0.1, head 1 sink = -0.2

Append sink column to HEAD 0:
           K[0]    K[1]    K[2]   SINK
  Q[0]: [ 0.670    -∞      -∞    0.1 ]
  Q[1]: [ 0.095  -0.132    -∞    0.1 ]
  Q[2]: [ 0.199   0.209  -0.354  0.1 ]

Shape: (1,2,3,3) → (1,2,3,4) — +1 column!

Softmax row Q[0]: [0.670, -∞, -∞, 0.1]
  e^0.670 = 1.954, e^0.1 = 1.105, sum = 3.059
  = [0.639, 0, 0, 0.361]   ← 36% absorbed by sink!

Softmax row Q[1]: [0.095, -0.132, -∞, 0.1]
  e^0.095=1.100, e^-0.132=0.876, e^0.1=1.105, sum=3.081
  = [0.357, 0.284, 0, 0.359]

DROP sink column:
           K[0]    K[1]    K[2]
  Q[0]: [ 0.639   0.000   0.000 ]   ← sums to 0.639 (not 1!)
  Q[1]: [ 0.357   0.284   0.000 ]   ← sums to 0.641
  Q[2]: [ 0.286   0.289   0.165 ]   ← sums to 0.741

Shape back to: (1, 2, 3, 3)
```

---

## Step 8: Weighted Sum of Values

```
V[0] = [0.28, 0.91], V[1] = [0.74, -0.15], V[2] = [-0.32, 0.58]

HEAD 0, position 0: weights = [0.639, 0, 0]
  output = 0.639 × [0.28, 0.91] = [0.179, 0.581]

HEAD 0, position 1: weights = [0.357, 0.284, 0]
  output = 0.357×[0.28,0.91] + 0.284×[0.74,-0.15]
         = [0.100, 0.325] + [0.210, -0.043] = [0.310, 0.282]

HEAD 0, position 2: weights = [0.286, 0.289, 0.165]
  output = 0.286×[0.28,0.91] + 0.289×[0.74,-0.15] + 0.165×[-0.32,0.58]
         = [0.080, 0.260] + [0.214, -0.043] + [-0.053, 0.096]
         = [0.241, 0.313]

Output shape: (3, 1, 2, 2)
```

---

## Step 9: Flatten + Output Projection + Residual

```
Reshape (3,1,2,2) → flatten → (3, 4)
  token 0: [0.179, 0.581, 0.195, 0.644]  ← [head0 | head1]

Output projection: (3,4) @ W_out(4,4) = (3,4)
Residual: output = x_original + attn_projected
  (3,4) + (3,4) = (3,4) — shape preserved!
```

---

## Step 10: MoE (Router → Experts)

```
Router: x_normed @ W_gate → (3,4) @ (4,2) = (3,2)

gate_scores = [ 0.978  -0.149 ]  ← token 0: Expert 0 wins
              [ 0.215   0.843 ]  ← token 1: Expert 1 wins
              [ 0.541   0.322 ]  ← token 2: Expert 0 wins

Expert 0 processes tokens 0 and 2:
  Up:     (2,4) @ W_up(4,8) = (2,8)
  SwiGLU: split 8 → gate(4) + linear(4), multiply → (2,4)
  Down:   (2,4) @ W_down(4,4) = (2,4)

Add residual: (3,4) + moe_out(3,4) = (3,4)
```

---

## Step 11: Unembedding → Predictions

```
Final RMSNorm: (3,4) → (3,4)
W_unembed: (4, 6) — maps hidden_size to vocab_size

logits = x_normed @ W_unembed: (3,4) @ (4,6) = (3, 6)

logits = [ 2.1   0.3  -1.0   0.8   0.1   3.5 ]  ← predicts token 5
         [ 0.5   1.2   0.3   0.8   0.1   0.9 ]  ← predicts token 1
         [ 0.9   0.6   4.2   0.1   0.5   0.8 ]  ← predicts token 2

Output shape: (3, 6) = (seq_len, vocab_size)
```

---

## Complete Shape Pipeline (26 Steps)

| Step | Operation | Matrix Multiply | Shape |
|------|-----------|----------------|-------|
| 1 | Input IDs | — | **(3,)** |
| 2 | Embedding lookup | table rows | **(3, 4)** |
| 3 | RMSNorm | element-wise | **(3, 4)** |
| 4 | QKV projection | (3,4)@(4,8) | **(3, 8)** |
| 5 | Split Q | slice | **(3, 4)** |
| 6 | Split K | slice | **(3, 2)** |
| 7 | Split V | slice | **(3, 2)** |
| 8 | Q reshape GQA | reshape | **(3, 1, 2, 2)** |
| 9 | Q·K^T | dot products | **(1, 2, 3, 3)** |
| 10 | Scale (1/√d) | element-wise | **(1, 2, 3, 3)** |
| 11 | + Causal mask | add | **(1, 2, 3, 3)** |
| 12 | + Sink column | concat | **(1, 2, 3, 4)** ← +1! |
| 13 | Softmax | per-row | **(1, 2, 3, 4)** |
| 14 | Drop sink | remove col | **(1, 2, 3, 3)** |
| 15 | Weights × Values | weighted sum | **(3, 1, 2, 2)** |
| 16 | Flatten heads | reshape | **(3, 4)** |
| 17 | Output projection | (3,4)@(4,4) | **(3, 4)** |
| 18 | + Residual | add | **(3, 4)** |
| 19 | RMSNorm | element-wise | **(3, 4)** |
| 20 | Router gate | (3,4)@(4,2) | **(3, 2)** |
| 21 | Expert up | (tok,4)@(4,8) | **(tok, 8)** |
| 22 | SwiGLU | gate×linear | **(tok, 4)** ← halved! |
| 23 | Expert down | (tok,4)@(4,4) | **(tok, 4)** |
| 24 | + Residual | add | **(3, 4)** |
| 25 | Final RMSNorm | element-wise | **(3, 4)** |
| 26 | Unembedding | (3,4)@(4,6) | **(3, 6)** |

---

## Key Patterns

- **Matrix multiply rule:** `(m, n) @ (n, p) = (m, p)` — inner dims match, outer dims form result
- **Shape invariant:** Between steps 2–24, shape is ALWAYS `(seq_len, hidden_size)`
- **Attention is O(n²):** Score matrix is `(seq × seq)`
- **Sink +1/-1:** Only place a dimension temporarily grows then shrinks
- **SwiGLU halves:** Expert up produces 2×, then gating cuts in half
- **Residual constrains design:** `x + sublayer(x)` forces sublayer to output same shape
