from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 201088
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 1024
    num_experts: int = 4
    experts_per_token: int = 2
    sliding_window: int = 128
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    context_len: int = 4096
    batch_size: int = 64
    grad_accum_steps: int = 8
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
