"""Entrenamiento reproducible desde pesos aleatorios; sin modelos externos."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from .data import random_batch, read_bytes, split_data
from .model import ByteTransformer, ModelConfig


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps, batch-size y eval-every deben ser >= 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    config = ModelConfig(context=args.context, d_model=args.d_model,
                         layers=args.layers, heads=args.heads, dropout=args.dropout)
    data = read_bytes(args.data)
    train_data, val_data = split_data(data)
    if min(len(train_data), len(val_data)) <= config.context:
        raise ValueError("Dataset insuficiente para el contexto configurado. Reducí --context o agregá datos.")
    model = ByteTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"Dispositivo={device}; parámetros={sum(p.numel() for p in model.parameters()):,}; bytes={len(data):,}")
    metrics = []
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = random_batch(train_data, args.batch_size, config.context, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = random_batch(val_data, args.batch_size, config.context, device)
                _, val_loss = model(vx, vy)
            entry = {"step": step, "train_loss": round(loss.item(), 4),
                     "val_loss": round(val_loss.item(), 4)}
            metrics.append(entry)
            print(f"Paso {step}/{args.steps} | train={entry['train_loss']} | val={entry['val_loss']}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint contiene únicamente tensores y tipos primitivos: lectura con weights_only=True.
    torch.save({"model_config": config.to_dict(), "state_dict": model.state_dict(),
                "step": args.steps, "seed": args.seed,
                "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Checkpoint creado: {output}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar MJON IA desde cero")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--out", default="checkpoints/demo.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
