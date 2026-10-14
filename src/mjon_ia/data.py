"""Preparación de corpus de prueba y muestreo; tokens = bytes UTF-8."""
import hashlib
import random
from pathlib import Path
import torch


def read_bytes(path: str | Path) -> torch.Tensor:
    content = Path(path).read_bytes()
    if len(content) < 100:
        raise ValueError('Dataset demasiado pequeño: deben existir >=100 bytes')
    return torch.tensor(list(content), dtype=torch.long)


def split_data(data: torch.Tensor, fraction: float = 0.9):
    """Partición por bytes heredada (no recomendable para datos reales)."""
    index = int(len(data) * fraction)
    if index < 2 or len(data) - index < 2:
        raise ValueError('No hay datos suficientes para separar entrenamiento y validación')
    return data[:index], data[index:]


def split_dialogues(text: str, fraction: float = 0.9, seed: int = 42):
    """Agrupar diálogos duplicados antes de particionar, sin mezclar bloques entre splits."""
    if not 0 < fraction < 1:
        raise ValueError('La fracción de entrenamiento debe estar entre cero y uno')
    blocks = [block.strip() for block in text.replace('\r\n', '\n').split('\n\n') if block.strip()]
    groups = {}
    for block in blocks:
        key = hashlib.sha256(block.encode('utf-8')).hexdigest()
        groups.setdefault(key, block)
    uniques = list(groups.values())
    if len(uniques) < 4:
        raise ValueError('Se requieren al menos 4 bloques de texto distintos separados por líneas en blanco')
    uniques.sort(key=lambda b: hashlib.sha256(b.encode('utf-8')).hexdigest())
    random.Random(seed).shuffle(uniques)
    index = min(len(uniques)-1, max(1, int(len(uniques) * fraction)))
    train = '\n\n'.join(uniques[:index]) + '\n\n'
    val = '\n\n'.join(uniques[index:]) + '\n\n'
    return (torch.tensor(list(train.encode('utf-8')), dtype=torch.long),
            torch.tensor(list(val.encode('utf-8')), dtype=torch.long))


def random_batch(data, batch_size, context, device):
    if len(data) <= context:
        raise ValueError('La partición de datos es menor al contexto (reducí --context)')
    offsets = torch.randint(0, len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in offsets.tolist()]).to(device)
    y = torch.stack([data[i+1:i+context+1] for i in offsets.tolist()]).to(device)
    return x,y
