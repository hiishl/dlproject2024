import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttn(nn.Module):
    """multihead causal self-attention
    multihead in parellel
    causal: mask the attention scores of tokens afterwards when caculating the attention weights matrix
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_emb % config.n_heads == 0

        # get the dim of each head
        self.head_dim = config.d_emb // config.n_heads
        # trainable weights to query, key and value
        self.W_q = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        # combine head outputs
        self.out_proj = nn.Linear(config.d_emb, config.d_emb)
        self.dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.block_size, config.block_size),
                       diagonal=1)
        )
        self.config = config

    def forward(self, x):
        b, n_tokens, d_emb = x.shape
        keys = self.W_k(x)       #(b, n_tokens, d_emb)
        queries = self.W_q(x)    #(b, n_tokens, d_emb)
        values = self.W_v(x)     #(b, n_tokens, d_emb)

        # split the matrix by adding a n_heads dimension
        keys = keys.view(b, n_tokens, self.config.n_heads, self.head_dim)
        values = values.view(b, n_tokens, self.config.n_heads, self.head_dim)
        queries = queries.view(b, n_tokens, self.config.n_heads, self.head_dim)
        # from (b, n_tokens, n_heads, head_dim) to (b, n_heads, n_tokens, head_dim)
        # so later can operate on the last two dimensions
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)

        # masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        # use the mask to fill the uper trianglular part to -inf,later can be softmaxed to 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # use softmax to standardize
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        # randomly drop out some entries in the attention weights matrix
        attn_weights = self.dropout(attn_weights) # (b, n_heads, n_tokens, n_tokens)

        # from (b, n_heads, n_tokens, head.dim) to (b, n_tokens, n_heads, head_dim)
        # for transforming back to the original dims
        context_vec = (attn_weights @ values).transpose(1, 2)
        # combines heads: d_out = n_heads * head_dim
        context_vec = context_vec.contiguous().view(
            b, n_tokens, d_emb
        )
        # add a linear layer
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    """normalize the inputs by layer rather than by batch
    independent of batch size, work well even single-instance batches
    stabilize training
    learnable parameters? 
    1. allow the model to recover the original distribution of the inputs if necessary
    2. control how much normalization affects the output
    """
    def __init__(self, config):
        super().__init__()
        # a small constant add to the var to avoid division by 0
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config.d_emb))
        self.shift = nn.Parameter(torch.zeros(config.d_emb))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # large sample size, no big difference in unbiased or not
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_emb, config.d_mlp*config.d_emb),
            nn.ReLU(),
            nn.Linear(config.d_mlp*config.d_emb, config.d_emb)
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)
        self.dropout = nn.Dropout(config.drop_rate)
        self.attn = CausalSelfAttn(config)
        self.feedfwd = MLP(config)

    def forward(self, x):
        # attention part
        # ResNet 1
        res = x
        # prenorm 1
        x = self.norm1(x) # (block_size, d_emb)
        x = self.attn(x) # (block_size, d_emb*n_heads)
        x = self.dropout(x)
        x = x + res
        # mlp part
        # ResNet 2
        res = x
        # prenorm 2
        x = self.norm2(x)
        x = self.feedfwd(x)
        x = self.dropout(x)
        x = res + x
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_emb)
        # positional embedding
        self.pos_emb = nn.Embedding(config.block_size, config.d_emb)
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = LayerNorm(config)
        self.out_head = nn.Linear(
            config.d_emb, config.vocab_size, bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

