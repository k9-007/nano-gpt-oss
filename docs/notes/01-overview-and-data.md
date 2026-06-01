# 01 — Overview, Dataset & Tokenization

## Why GPT-OSS?

GPT-OSS is a **truly open-source** GPT implementation that goes beyond just releasing weights. It provides the complete pipeline: dataset preparation, tokenizer training, architecture, pre-training, and inference — all designed to be studied and reproduced.

### Key Differences from GPT-2

| Innovation | GPT-2 | GPT-OSS | Why |
|-----------|--------|---------|-----|
| Normalization | LayerNorm | RMSNorm | ~15% faster, skips mean |
| Position Encoding | Absolute (additive) | RoPE (rotation) | No semantic pollution |
| Attention | MHA (all unique) | GQA (4:1 sharing) | 4× smaller KV cache |
| Attention Window | Full causal | Alternating full + sliding | ~1000× fewer ops |
| Activation | GELU | SwiGLU | Better gradient flow |
| FFN | Dense | Mixture of Experts (MoE) | More capacity, same compute |
| Position Awareness | Fixed 1024 tokens | YaRN-extended RoPE | Generalizes beyond training length |

---

## Dataset: TinyStories

**Why TinyStories?**
- Contains simple but grammatically correct English stories
- Teaches the model language structure, grammar, and basic reasoning
- Small enough to train in reasonable time on a single GPU
- Large enough (10M+ stories) to learn meaningful representations

**Properties:**
- Vocabulary covers common English words
- Stories have clear narrative structure (beginning, middle, end)
- Sentences are short and well-formed
- Good for learning: pronouns, tenses, conjunctions, basic logic

---

## Tokenization: Characters → Words → Subwords → BPE

### The Evolution

| Method | Example: "unhappiness" | Problem |
|--------|----------------------|---------|
| Character-level | `u, n, h, a, p, p, i, n, e, s, s` | Too many tokens, no meaning per token |
| Word-level | `unhappiness` | OOV (out-of-vocabulary) for rare words |
| **Subword (BPE)** | `un, happi, ness` | Best of both: covers all words, meaningful pieces |

### Byte Pair Encoding (BPE) — How It Works

1. Start with individual characters as tokens
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size

```
Corpus: "low lower lowest"

Step 1: Characters = {l, o, w, e, r, s, t, ' '}
Step 2: Most frequent pair = ('l', 'o') → merge into 'lo'
Step 3: Most frequent pair = ('lo', 'w') → merge into 'low'
Step 4: Continue...

Final tokens: {"low", "er", "est", ...}
```

### Harmony Tokenizer (GPT-OSS)

Built on top of OpenAI's O200K BPE base, extended with special tokens:

```
Base vocabulary: 200,000 tokens (O200K BPE)
+ Special tokens for chat/roles:
  <|begin_of_text|>    — start of sequence
  <|end_of_text|>      — end of sequence
  <|start_header|>     — role marker start
  <|end_header|>       — role marker end
  <|tool_call|>        — agentic channels
  ...

Total: 201,088 tokens
```

**Why extend O200K?**
- O200K already has excellent English subword coverage
- Adding special tokens enables chat formatting and tool use
- Larger vocab = shorter sequences = faster training (each token carries more meaning)

---

## Creating Input-Output Pairs

GPT-OSS uses **next-token prediction** (auto-regressive self-supervised learning):

```
Original text: "The cat sat on the mat"
Tokenized:     [The, cat, sat, on, the, mat]

Input:  [The, cat, sat, on,  the]  ← positions 0-4
Target: [cat, sat, on,  the, mat]  ← positions 1-5 (shifted by 1)

At each position, the model predicts what comes NEXT.
```

**Why this works:**
- No manual labeling needed — the text IS the label (shifted by 1)
- Forces the model to learn grammar, semantics, world knowledge
- Scales to unlimited data (any text becomes training data)
- The causal mask ensures the model never "cheats" by looking ahead

### Data Loading Pipeline

```
Raw text → Tokenize → Split into chunks of context_length
         → Create (input, target) pairs where target = input shifted by 1
         → Batch into groups of batch_size
         → Feed to model
```
