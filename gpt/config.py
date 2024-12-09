from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 128
    batch_size: int = 128
    d_emb: int = 768
    n_heads: int = 8
    n_layers: int = 12
    drop_rate: float = 0.1
    d_mlp: int = 4
    qkv_bias: bool = True
    vocab_size: int = None