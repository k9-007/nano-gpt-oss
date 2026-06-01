# 07 — Code Walkthrough

## tokenizer.py

```python
import tiktoken

class HarmonyTokenizer:
    """
    Extends OpenAI's O200K BPE with special tokens for chat and tool use.
    Total vocab: 201,088 = 200,000 (O200K base) + 1,088 (special tokens)
    """
    def __init__(self):
        # Load O200K base (same base as GPT-4)
        base = tiktoken.get_encoding("o200k_base")
        
        # Define special tokens with unique IDs starting after base vocab
        special_tokens = {
            "<|begin_of_text|>": 200000,
            "<|end_of_text|>": 200001,
            "<|start_header|>": 200002,
            "<|end_header|>": 200003,
            # ... more special tokens for roles, tools, etc.
        }
        
        # Create extended encoding
        self.encoding = tiktoken.Encoding(
            name="harmony",
            pat_str=base._pat_str,       # same regex pattern
            mergeable_ranks=base._mergeable_ranks,  # same BPE merges
            special_tokens=special_tokens
        )
    
    def encode(self, text):
        """Text → list of token IDs"""
        return self.encoding.encode(text, allowed_special="all")
    
    def decode(self, ids):
        """List of token IDs → text"""
        return self.encoding.decode(ids)
    
    @property
    def vocab_size(self):
        return 201088
    
    @property
    def bos_id(self):  # beginning of sequence
        return 200000
    
    @property
    def eos_id(self):  # end of sequence
        return 200001
```

---

## data_loader.py

```python
import torch
import numpy as np

class DataLoader:
    """
    Loads pre-tokenized data and creates (input, target) pairs
    for next-token prediction.
    
    Data format: .bin files containing uint32 token IDs
    """
    def __init__(self, data_path, context_len, batch_size):
        self.context_len = context_len
        self.batch_size = batch_size
        
        # Memory-map the data file (doesn't load into RAM)
        # This is critical for large datasets that don't fit in memory
        self.data = np.memmap(data_path, dtype=np.uint32, mode='r')
        self.n_tokens = len(self.data)
        self.position = 0  # current read position
    
    def get_batch(self):
        """
        Returns (input_ids, targets) each of shape (batch_size, context_len)
        Targets are just input shifted by 1 position.
        """
        batch_inputs = []
        batch_targets = []
        
        for _ in range(self.batch_size):
            # Grab a chunk of context_len + 1 tokens
            chunk = self.data[self.position : self.position + self.context_len + 1]
            
            # Input = first context_len tokens
            # Target = last context_len tokens (shifted by 1)
            batch_inputs.append(chunk[:-1])   # [0, 1, ..., ctx-1]
            batch_targets.append(chunk[1:])   # [1, 2, ..., ctx]
            
            self.position += self.context_len
            if self.position + self.context_len + 1 > self.n_tokens:
                self.position = 0  # wrap around
        
        inputs = torch.tensor(np.array(batch_inputs), dtype=torch.long)
        targets = torch.tensor(np.array(batch_targets), dtype=torch.long)
        return inputs, targets
```

---

## gptoss.py (Architecture)

### ModelConfig

```python
@dataclass
class ModelConfig:
    vocab_size: int = 201088
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16       # Q heads
    num_key_value_heads: int = 4        # KV heads (GQA 4:1)
    head_dim: int = 64                  # hidden_size / num_attention_heads
    intermediate_size: int = 1024       # expert hidden dim
    num_experts: int = 4
    experts_per_token: int = 2
    sliding_window: int = 128
    rope_theta: float = 500000.0        # RoPE base frequency
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
```

### Main Model

```python
class GPTOSS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        
        self.final_norm = RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)          # (seq,) → (seq, hidden)
        
        for block in self.blocks:
            x = block(x)                       # (seq, hidden) → (seq, hidden)
        
        x = self.final_norm(x)                 # (seq, hidden) → (seq, hidden)
        logits = self.output(x)                # (seq, hidden) → (seq, vocab)
        return logits
```

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attention = AttentionBlock(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.moe = MoEBlock(config)
    
    def forward(self, x):
        # Pre-Norm + Attention + Residual
        x = x + self.attention(self.attn_norm(x))
        # Pre-Norm + MoE + Residual
        x = x + self.moe(self.ffn_norm(x))
        return x
```

### Attention Block

```python
class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Attention sinks: one learnable scalar per Q head
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))
        
        # Sliding window for even layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        
        # Pre-compute RoPE frequencies
        self.freqs = precompute_rope_freqs(config)
    
    def forward(self, x):
        seq_len = x.shape[0]
        
        # QKV projection
        qkv = self.qkv(x)  # (seq, hidden) → (seq, qkv_dim)
        q, k, v = split_qkv(qkv)
        
        # Apply RoPE to Q and K
        q = apply_rope(q, self.freqs[:seq_len])
        k = apply_rope(k, self.freqs[:seq_len])
        
        # Reshape for GQA
        q = q.view(seq_len, num_kv_heads, q_per_group, head_dim)
        k = k.view(seq_len, num_kv_heads, head_dim)
        v = v.view(seq_len, num_kv_heads, head_dim)
        
        # Attention: Q·K^T, scale, mask, sinks, softmax, ×V
        output = sdpa_with_sinks(q, k, v, self.sinks, self.sliding_window)
        
        # Flatten and project
        output = output.reshape(seq_len, -1)
        return self.out_proj(output)
```

### MoE Block

```python
class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(config) for _ in range(config.num_experts)
        ])
        self.experts_per_token = config.experts_per_token
    
    def forward(self, x):
        # Route
        scores = self.router(x)  # (seq, num_experts)
        weights, indices = scores.topk(self.experts_per_token)
        weights = F.softmax(weights, dim=-1)
        
        # Dispatch to experts and blend
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)  # which tokens go here
            if mask.any():
                expert_out = expert(x[mask])
                # Weight by routing score
                token_weights = weights[mask, (indices[mask] == i).nonzero()[:, 1]]
                output[mask] += expert_out * token_weights.unsqueeze(-1)
        
        return output
```

---

## trainer.py

```python
def train(config):
    model = GPTOSS(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)
    dataloader = DataLoader(config.data_path, config.context_len, config.batch_size)
    
    for step in range(config.max_steps):
        lr = get_cosine_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.zero_grad()
        total_loss = 0
        
        for _ in range(config.grad_accum_steps):
            inputs, targets = dataloader.get_batch()
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs.view(-1))
            loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                    targets.view(-1))
            loss = loss / config.grad_accum_steps
            loss.backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 5000 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'step': step,
            }, f'checkpoint_{step}.pt')
```

---

## inference.py

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200,
             temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).to(device)
    
    for _ in range(max_tokens):
        # Only use last context_len tokens if sequence is too long
        context = input_ids[-config.max_position_embeddings:]
        
        logits = model(context)
        next_logits = logits[-1] / temperature  # last position, scaled
        
        # Top-K filtering
        top_k_logits, top_k_indices = next_logits.topk(top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        
        # Top-P filtering
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = (cumulative - sorted_probs) >= top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        
        # Sample
        idx = torch.multinomial(sorted_probs, 1)
        next_token = top_k_indices[sorted_idx[idx]]
        
        if next_token.item() == tokenizer.eos_id:
            break
        
        input_ids = torch.cat([input_ids, next_token])
    
    return tokenizer.decode(input_ids.tolist())
```
