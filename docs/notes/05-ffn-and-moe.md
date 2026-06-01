# 05 — FFN, SwiGLU & Mixture of Experts

## Activation Function Evolution

### Why Not Just ReLU?

```
ReLU(x) = max(0, x)

Problem: "dying neurons" — once a neuron goes negative, gradient = 0 forever.
Also: hard cutoff at 0 loses information about slightly negative values.
```

### The Evolution

| Function | Formula | Advantage |
|----------|---------|-----------|
| ReLU | `max(0, x)` | Simple, fast |
| GELU | `x × Φ(x)` (Gaussian CDF) | Smoother, no dying neurons |
| GLU | `σ(xW₁) × xW₂` | Gating mechanism |
| **SwiGLU** | `swish(xW_gate) × (xW_linear + 1)` | Best gradient flow + gating |

### GLU (Gated Linear Unit) — The Key Insight

Instead of one path through the FFN, use TWO paths:
- **Gate path:** decides HOW MUCH information flows
- **Linear path:** carries the actual information

```
GLU(x) = gate(x) × linear(x)
       = activation(x @ W_gate) × (x @ W_linear)
```

### SwiGLU — Best of All Worlds

```python
# SwiGLU combines Swish activation with GLU gating
def swiglu(x, W_up):
    up = x @ W_up              # (seq, hidden) @ (hidden, 2*intermediate)
    gate = up[..., ::2]        # even indices → gate path
    linear = up[..., 1::2]     # odd indices → linear path
    return swish(gate) * (linear + 1.0)

# Where swish(x) = x × sigmoid(x) — smooth, non-monotonic
```

**Why SwiGLU > GELU:**
- Gating lets the network learn what to pass through and what to block
- `+1` in the linear path ensures non-zero gradient even when linear ≈ 0
- Swish is smoother than sigmoid, better gradient flow
- Empirically trains faster and achieves lower loss

### Dimension Flow in SwiGLU

```
Input:  (seq, hidden_size) = (4096, 1024)
W_up:   (hidden_size, 2 × intermediate) = (1024, 2048)
After matmul: (4096, 2048)
Split:  gate = (4096, 1024), linear = (4096, 1024)
SwiGLU: (4096, 1024) — halved!
W_down: (intermediate, hidden_size) = (1024, 1024)
Output: (4096, 1024) — back to original size
```

---

## Mixture of Experts (MoE)

### Why Not One Big FFN?

A single large FFN:
- Uses ALL parameters for EVERY token (wasteful)
- Limited capacity at fixed compute budget

MoE solution:
- Multiple specialized FFNs (experts)
- Each token only uses top-K experts
- 4× more parameters, same compute cost!

### Architecture

```
Input token → Router → scores for each expert
           → Top-K selection (K=2 for GPT-OSS)
           → Route to selected experts only
           → Weighted blend of expert outputs
           → Output
```

### The Router

```python
class Router(nn.Module):
    def __init__(self, hidden_size, num_experts):
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        # Score each expert for each token
        scores = self.gate(x)  # (seq, hidden) @ (hidden, num_experts)
        # Top-K selection
        top_k_scores, top_k_indices = scores.topk(k=experts_per_token)
        # Softmax over selected experts only
        weights = F.softmax(top_k_scores, dim=-1)
        return weights, top_k_indices
```

### Expert Specialization

After training, experts naturally specialize:
```
Expert 0: handles syntax patterns ("the", "is", articles)
Expert 1: handles entities and nouns
Expert 2: handles verbs and actions
Expert 3: handles descriptive/adjective patterns
```

The router learns which expert is best for each token type.

### Load Balancing

Without balancing, the router might always pick the same expert (collapse):

```python
# Auxiliary loss encourages balanced routing
# f_i = fraction of tokens routed to expert i (want ≈ 1/num_experts)
# P_i = average routing probability for expert i
load_balance_loss = num_experts * sum(f_i * P_i)
```

### GPT-OSS MoE Config

```
num_experts = 4        # total available experts
experts_per_token = 2  # each token activates 2 experts
intermediate_size = 1024  # expert hidden dim

Active parameters per token: 2/4 = 50% of expert capacity
Total parameters: 4× a single FFN
Compute: same as 2 FFNs (only 2 are active)
```

### Why 2 Experts Per Token?

- 1 expert: too brittle, single point of failure
- 2 experts: diversity + redundancy, smooth routing
- 3+ experts: diminishing returns, more compute

---

## Complete FFN Block Flow

```
x (3, 4)
  → RMSNorm
  → Router: (3,4) @ W_gate(4, num_experts) = (3, num_experts)
  → Top-K selection
  → For each selected expert:
      → Expert_up:   (tokens, 4) @ W_up(4, 8) = (tokens, 8)
      → SwiGLU:      (tokens, 8) → (tokens, 4)
      → Expert_down: (tokens, 4) @ W_down(4, 4) = (tokens, 4)
  → Weighted blend of expert outputs
  → + Residual
= x_out (3, 4)  ← same shape!
```
