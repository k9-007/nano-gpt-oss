# GPT-OSS Architecture Notes

Comprehensive study notes for the Nano GPT-OSS architecture — covering every design decision from tokenization to inference.

## Notes Index

| # | File | Topics |
|---|------|--------|
| 1 | [01-overview-and-data.md](./01-overview-and-data.md) | Motivation, TinyStories dataset, Tokenization (BPE → Harmony) |
| 2 | [02-dimensions-guide.md](./02-dimensions-guide.md) | Complete matrix shape walkthrough with actual arithmetic |
| 3 | [03-normalization-and-embeddings.md](./03-normalization-and-embeddings.md) | Token embeddings, RMSNorm vs LayerNorm |
| 4 | [04-attention-mechanisms.md](./04-attention-mechanisms.md) | MHA → MQA → GQA, Sliding Window, RoPE, Attention Sinks |
| 5 | [05-ffn-and-moe.md](./05-ffn-and-moe.md) | GLU → SwiGLU evolution, Mixture of Experts, Router |
| 6 | [06-training-and-inference.md](./06-training-and-inference.md) | Training pipeline, AdamW, Cosine LR, Inference sampling |
| 7 | [07-code-walkthrough.md](./07-code-walkthrough.md) | Full code explanation: tokenizer, data_loader, model, trainer, inference |
| 8 | [08-deep-questions.md](./08-deep-questions.md) | Deep architectural questions and detailed answers |

## Interactive Version

For diagrams and interactive elements, see the [HTML notes](./gpt-oss-notes.html).

## Based On

- [VizuaraAI/nano-gpt-oss](https://github.com/VizuaraAILabs/nano-gpt-oss)
- [Video Lecture (3hr Deep Dive)](https://www.youtube.com/watch?v=hBUsySdcA3I)
