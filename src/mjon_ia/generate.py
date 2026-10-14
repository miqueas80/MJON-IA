"""Generador CLI para checkpoints entrenados en este proyecto."""
import argparse

import torch

from .model import ByteTransformer, ModelConfig


def run(args):
    torch.set_num_threads(2)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = ByteTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    encoded = args.prompt.encode("utf-8")
    if not encoded:
        raise ValueError("El prompt no puede estar vacío")
    x = torch.tensor([list(encoded)], dtype=torch.long, device=args.device)
    out = model.generate(x, new_tokens=args.tokens, temperature=args.temperature,
                         top_k=args.top_k)[0].tolist()
    # Algunos bytes pueden formar secuencias UTF-8 incompletas en un modelo pequeño.
    print(bytes(out).decode("utf-8", errors="replace"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generación local usando pesos propios")
    parser.add_argument("--model", default="checkpoints/demo.pt")
    parser.add_argument("--prompt", default="Usuario: Hola\nAsistente:")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
