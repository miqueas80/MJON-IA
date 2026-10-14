"""Transformer autoregresivo pequeño: todas las capas se entrenan desde cero."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256  # tokenización de bytes UTF-8; sin vocabulario externo
    context: int = 128
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.context < 2 or self.d_model < 1 or self.layers < 1 or self.heads < 1:
            raise ValueError("Las dimensiones deben ser positivas y context >= 2")
        if self.d_model % self.heads:
            raise ValueError("d_model debe ser divisible por heads")
        if self.vocab_size != 256:
            raise ValueError("El tokenizer de este prototipo requiere vocab_size=256")

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, seq, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t):
            return t.view(batch, seq, self.n_heads, width // self.n_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            reshape(q), reshape(k), reshape(v), is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq, width)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ByteTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.context, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        batch, seq = tokens.shape
        if seq > self.config.context:
            raise ValueError("La secuencia supera el contexto configurado")
        pos = torch.arange(seq, device=tokens.device)
        x = self.drop(self.token(tokens) + self.position(pos))
        x = self.norm(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, start: torch.Tensor, new_tokens=100, temperature=0.9, top_k=40):
        if temperature <= 0:
            raise ValueError("temperature debe ser > 0")
        self.eval()
        generated = start
        for _ in range(new_tokens):
            logits, _ = self(generated[:, -self.config.context:])
            logits = logits[:, -1, :] / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
