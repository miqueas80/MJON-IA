"""MJON IA Neural Core: Transformer causal propio, escalable y entrenable.

Implementación independiente de pesos de terceros: RoPE + RMSNorm + SwiGLU +
Grouped Query Attention + PyTorch SDPA. Los modelos legacy v0.1-v1.2
conservan src/mjon_ia/model.py y su ejecución web sin cambios.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NeuralConfig:
    vocab_size: int = 4096
    context: int = 1024
    width: int = 384
    layers: int = 8
    heads: int = 6
    kv_heads: int = 2
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.vocab_size < 256 or self.context < 2 or self.width < 16:
            raise ValueError("Vocabulario, contexto o ancho inválidos")
        if self.layers < 1 or self.heads < 1 or self.kv_heads < 1:
            raise ValueError("Capas y cabezas deben ser positivas")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width debe ser divisible por heads, y heads por kv_heads")
        if (self.width // self.heads) % 2:
            raise ValueError("RoPE requiere dimensión par por cabeza")
        if not 0 <= self.dropout < 1 or self.rope_theta <= 1:
            raise ValueError("Dropout o frecuencia RoPE inválidos")

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(norm + self.eps)).to(x.dtype) * self.weight


def apply_rope(x, position, theta: float):
    head_dim = x.shape[-1]
    freq = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inv = theta ** (-freq / head_dim)
    angles = torch.outer(position.float(), inv)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype=x.dtype)[None,None,:,:]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype=x.dtype)[None,None,:,:]
    a, b = x.chunk(2, dim=-1)
    rotate = torch.cat((-b, a), dim=-1)
    return (x * cos) + (rotate * sin)


class GroupedAttention(nn.Module):
    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.width // config.heads
        self.theta = config.rope_theta
        self.dropout = config.dropout
        self.q = nn.Linear(config.width, config.heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, past=None, use_cache=False):
        batch, seq, _ = x.shape
        previous = 0 if past is None else past[0].shape[2]
        def heads(t, count):
            return t.view(batch, seq, count, self.head_dim).transpose(1, 2)
        position = torch.arange(previous, previous + seq, device=x.device)
        query = apply_rope(heads(self.q(x), self.heads), position, self.theta)
        key = apply_rope(heads(self.k(x), self.kv_heads), position, self.theta)
        value = heads(self.v(x), self.kv_heads)
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            value = torch.cat((past[1], value), dim=2)
        next_cache = (key, value) if use_cache else None
        if self.heads != self.kv_heads:
            ratio = self.heads // self.kv_heads
            key = key.repeat_interleave(ratio, dim=1)
            value = value.repeat_interleave(ratio, dim=1)
        attn_mask = None
        causal = past is None
        if past is not None:
            # At decoding time, each new token can see every cached past token.
            # Also supports chunked cache with multiple incoming tokens.
            kpos = torch.arange(key.shape[2], device=x.device)
            qpos = previous + torch.arange(seq, device=x.device)
            attn_mask = kpos[None, :] <= qpos[:, None]
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        result = result.transpose(1,2).contiguous().view(batch,seq,-1)
        return self.proj(result), next_cache


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = 64 * math.ceil((8 * width / 3) / 64)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class NeuralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.width)
        self.attn = GroupedAttention(config)
        self.mlp_norm = RMSNorm(config.width)
        self.mlp = SwiGLU(config.width)

    def forward(self, x, past=None, use_cache=False):
        a, cache = self.attn(self.attn_norm(x), past=past, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache


class MJONNeuralCore(nn.Module):
    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.layers = nn.ModuleList(NeuralBlock(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.output.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, (nn.Embedding, nn.Linear)):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("Se requieren tokens [batch, sequence]")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("Los tokens deben ser enteros")
        cached = past_key_values
        if cached is not None and len(cached) != len(self.layers):
            raise ValueError("Longitud de KV cache incompatible")
        offset = 0 if cached is None else cached[0][0].shape[2]
        if offset + tokens.shape[1] > self.config.context:
            raise ValueError("Se superó el contexto de MJON")
        x = self.token(tokens)
        next_cache = []
        for index, layer in enumerate(self.layers):
            x, layer_cache = layer(
                x,
                past=None if cached is None else cached[index],
                use_cache=use_cache,
            )
            if use_cache:
                next_cache.append(layer_cache)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("targets y tokens deben compartir dimensiones")
            loss = F.cross_entropy(
                logits.float().reshape(-1,logits.shape[-1]),
                targets.reshape(-1), ignore_index=-100,
            )
        return logits, loss, tuple(next_cache) if use_cache else None

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64, eos_id=None, temperature=0.0):
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
            raise ValueError("La generación usa un prompt no vacío de un elemento")
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError("Parámetros de generación inválidos")
        self.eval()
        ids = prompt.clone()
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                # When context is full, reprocess the latest tokens using
                # fresh zero-based RoPE positions, never retain stale cache.
                x = ids[:, -self.config.context:]
            else:
                x = ids[:, -1:]
            if cache is not None and cache[0][0].shape[2] + 1 > self.config.context:
                cache = None
                x = ids[:, -self.config.context:]
            logits, _, cache = self(x, past_key_values=cache, use_cache=True)
            next_logits = logits[:, -1, :].float()
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1,keepdim=True)
            else:
                next_token = torch.multinomial(
                    F.softmax(next_logits / temperature,dim=-1),1)
            ids = torch.cat([ids,next_token],dim=1)
            if eos_id is not None and int(next_token.item()) == eos_id:
                break
        return ids

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
