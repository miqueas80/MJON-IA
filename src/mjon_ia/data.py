"""Dataset de demostración. Cada carácter se codifica como bytes UTF-8."""
from pathlib import Path

import torch


def read_bytes(path: str | Path) -> torch.Tensor:
    content = Path(path).read_bytes()
    if len(content) < 100:
        raise ValueError("Dataset demasiado pequeño: deben existir >=100 bytes")
    return torch.tensor(list(content), dtype=torch.long)


def split_data(data: torch.Tensor, fraction: float = 0.9):
    index = int(len(data) * fraction)
    if index < 2 or len(data) - index < 2:
        raise ValueError("No hay datos suficientes para separar entrenamiento y validación")
    return data[:index], data[index:]


def random_batch(data, batch_size, context, device):
    if len(data) <= context:
        raise ValueError("La partición de datos es menor al contexto (reducí --context)")
    offsets = torch.randint(0, len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in offsets.tolist()]).to(device)
    y = torch.stack([data[i+1:i+context+1] for i in offsets.tolist()]).to(device)
    return x, y
