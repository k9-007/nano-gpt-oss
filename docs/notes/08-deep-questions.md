# 08 — Deep Architectural Questions

## Why Different Q if K,V are Shared? (GQA)

**The Question:** If multiple Q heads share the same K and V, won't they see the same thing? Why keep multiple Q heads?

**The Answer:** Different Q = different WAY of asking the same memory.

### Concrete Example

```
Shared K = [[1, 0], [0, 1]]    # 2 tokens in memory
Shared V = [[10, 0], [0, 20]]  # their values

Q_head0 = [1, 0]  → focuses on token 0
Q_head1 = [0, 1]  → focuses on token 1

Head 0: softmax(Q0·K^T) = [1, 0] → output = V[0] = [10, 0]
Head 1: softmax(Q1·K^T) = [0, 1] → output = V[1] = [0, 20]
```

**Same memory, different questions → different answers!**

### Analogy: Library

- K, V = books in a library (shared knowledge)
- Q heads = different people with different questions

Even with the same library:
- Head 0 asks: "What's the syntax?"
- Head 1 asks: "What's the semantics?"
- Head 2 asks: "What's the long-range reference?"

### The Deep Insight

Attention is **asymmetric**:
- K, V = "what exists" (memory) → can be shared
- Q = "what I want" (intent) → MUST be different

You can share memory, but you cannot share intent.

---

## Why Rotate Instead of Add for Position? (RoPE)

**The Question:** Why is rotation better than adding a position vector?

### Problems with Additive Position

```
token = [0.8, 0.3, -0.5, 0.2]  (meaning: "cat")
pos_5 = [0.1, -0.2, 0.4, 0.1]  (position 5 embedding)

result = [0.9, 0.1, -0.1, 0.3]  ← this is neither "cat" nor "position 5"!
```

The semantic meaning is **contaminated** by position data.

### Why Rotation is Better

1. **Preserves magnitude:** `|rotated_vector| = |original_vector|` (meaning unchanged)
2. **Changes direction only:** position info encoded in angle, not size
3. **Relative position emerges naturally:** Q·K^T depends on angle DIFFERENCE
4. **No learned parameters:** just math (sine/cosine), works for any sequence length

### Mathematical Intuition

```
Rotation by angle θ:
  [cos θ  -sin θ] [x]   Length of result = length of [x, y]
  [sin θ   cos θ] [y]   Direction changed by θ

Q at pos m: rotated by mθ
K at pos n: rotated by nθ
Q·K = f(m-n) — depends only on RELATIVE distance!
```

---

## Why Remove the Mean Subtraction? (RMSNorm)

**The Question:** LayerNorm subtracts the mean to "center" the distribution. Why is that unnecessary?

### The Argument for Removing It

1. **Redundancy:** The bias term in the next linear layer can learn any centering needed
2. **Speed:** Removing mean computation saves one pass over the data
3. **Empirical results:** Training quality is the same or better
4. **Simplicity:** One fewer moving part = fewer things to debug

### Why It Still Works

- The model has billions of parameters to learn any needed shift
- What matters is the RELATIVE magnitudes between dimensions
- RMS division normalizes the scale without changing the center
- If centering were truly important, we'd see degraded training — we don't

---

## Why Gate the Activation Function? (SwiGLU)

**The Question:** Why is `gate(x) × linear(x)` better than just `activation(x)`?

### The Problem with Simple Activations

```
GELU: output = x × Φ(x)

Every dimension is processed identically — no selectivity.
The network can't learn "pass dimension 3 through but block dimension 7."
```

### The Gating Advantage

```
SwiGLU: output = swish(x @ W_gate) × (x @ W_linear + 1)

gate path → LEARNS which dimensions are important for this input
linear path → carries the actual information
multiplication → selective filtering
```

This gives the network explicit control over information flow:
- Gate ≈ 0 → block this information
- Gate ≈ 1 → pass this through
- Gate = 0.5 → attenuate (reduce strength)

### Why +1 in the Linear Path?

```
(x @ W_linear + 1)  instead of  (x @ W_linear)
```

Without +1: if W_linear output ≈ 0, gradient through gate × 0 = 0 (dead!)
With +1: even when W_linear ≈ 0, the term is ≈ 1, so gradient flows through gate.

---

## Why Not One Big FFN? (MoE)

**The Question:** Why use 4 small experts instead of 1 large FFN?

### The Compute-Capacity Tradeoff

```
Single FFN:    1024 → 4096 → 1024    (4× hidden expansion)
               ALL 4096 neurons active for EVERY token
               Parameters: 1024×4096 + 4096×1024 = 8.4M

4 Experts:     Each: 1024 → 1024 → 1024  (same hidden per expert)
               Only 2 experts active per token (top-2)
               Total params: 4 × (1024×1024 + 1024×1024) = 8.4M
               Active params: 2 × (1024×1024 + 1024×1024) = 4.2M

Same total params, 2× less compute per token!
OR: same compute, 2× more parameters (capacity)!
```

### Specialization

After training, experts naturally diverge:
- Expert 0 might handle function words
- Expert 1 might handle nouns/entities
- Expert 2 might handle verbs/actions
- Expert 3 might handle descriptors

This is emergent — not programmed!

---

## Why Do Attention Sinks Exist?

**The Question:** Why does the model need a "garbage dump" for attention?

### The Forced Allocation Problem

Softmax MUST sum to 1.0 — every query position MUST assign its full attention somewhere.

But what if a token genuinely doesn't need information from any past token?
- In early layers: fine (every token learns its own representation)
- In deep layers: some heads specialize and don't need to attend meaningfully

Without sinks: forced to put weight on irrelevant tokens → noisy outputs.
With sinks: excess attention goes to sink → remaining weights are meaningful.

### Cost: Just 16 Scalars

The entire mechanism costs only `num_attn_heads = 16` parameters for the whole model. Negligible cost, significant quality improvement.

---

## Why Residual Connections?

**The Question:** Why add the input back to the output?

### The Gradient Highway

Without residuals in a 24-layer network:
```
gradient = ∂L/∂x₀ = ∂L/∂x₂₃ × ∂x₂₃/∂x₂₂ × ... × ∂x₁/∂x₀
                   = product of 24 Jacobians → vanishes or explodes!
```

With residuals: `x_out = x_in + sublayer(x_in)`
```
∂x_out/∂x_in = I + ∂sublayer/∂x_in
             ↑ identity matrix! Gradient flows directly.
```

The gradient can flow unchanged through the identity path, even if the sublayer gradient vanishes.

### The Information Preservation Argument

- Each block only ADDS information (never destroys)
- Original token meaning is preserved in the residual stream
- Attention adds context, MoE adds knowledge — both additive

---

## Why Pre-Norm Over Post-Norm?

**Post-Norm (GPT-2):** `x → sublayer → + residual → norm`
**Pre-Norm (GPT-OSS):** `x → norm → sublayer → + residual`

### Why Pre-Norm Wins

1. **Stable residual stream:** The unnormalized residual stays clean
2. **Better gradients:** Norm before sublayer means sublayer inputs are well-scaled
3. **Deeper networks:** Post-norm diverges beyond ~12 layers without careful tuning
4. **No warmup tricks needed:** Pre-norm is stable from step 1

---

## Why Cosine LR Schedule?

**The Question:** Why not just use a constant learning rate?

### The Exploration-Refinement Tradeoff

```
Early training:  Loss landscape is bumpy, need big steps to escape bad regions
Late training:   Near a good minimum, need small steps for precision

Constant LR: either too big (overshoots at end) or too small (stuck at start)
Step decay:   sudden drops cause training instability
Cosine:       smooth transition from exploration → refinement
```

### Why Cosine Specifically?

- Smooth (no discontinuities that cause loss spikes)
- Spends more time at lower LR (refinement phase is longer)
- Mathematically: `0.5 × (1 + cos(π × progress))` naturally goes from 1→0
- Empirically matches the "diminishing returns" nature of training
