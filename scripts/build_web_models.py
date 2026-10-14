"""Entrena y exporta dos modelos pequeños desde pesos aleatorios para GitHub Pages."""
import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch

from mjon_ia.train import run as train_model
from export_web import run as export_model


VARIANTS = (
    dict(id="v01", label="v0.1 · modelo mínimo", description="Transformer original, 8.176 parámetros, 48 bytes de contexto", steps=800, context=48, d_model=16, layers=1, heads=2, batch_size=16, split_mode="bytes"),
    dict(id="v02", label="v0.2 · modelo ampliado", description="Transformer más grande y entrenamiento reanudable, 96 bytes de contexto", steps=900, context=96, d_model=32, layers=2, heads=4, batch_size=8, split_mode="dialogues"),
)


def build(output_dir="docs/web", dataset="data/demo.txt", checkpoint_dir="checkpoints/web"):
    output_dir, checkpoint_dir = Path(output_dir), Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    catalog = {"format": "mjonia.models.v1", "models": []}
    for v in VARIANTS:
        path = checkpoint_dir / f"{v['id']}.pt"
        train_model(Namespace(
            data=str(dataset), out=str(path), resume=None,
            steps=v["steps"], eval_every=150, save_every=150,
            split_mode=v["split_mode"], batch_size=v["batch_size"],
            context=v["context"], d_model=v["d_model"], layers=v["layers"],
            heads=v["heads"], dropout=0.0, lr=0.001,
            seed=42, cpu_threads=2, device="cpu",
        ))
        weights_path = output_dir / f"weights-{v['id']}.txt"
        export_model(str(path), str(weights_path))
        state = torch.load(path, weights_only=True, map_location="cpu")
        unique_storages = set()
        parameters = 0
        for tensor in state["state_dict"].values():
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.numel())
            if key not in unique_storages:
                unique_storages.add(key)
                parameters += tensor.numel()
        catalog["models"].append({
            "id": v["id"], "label": v["label"],
            "description": v["description"], "file": weights_path.name,
            "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "steps": state["step"], "context": v["context"],
            "parameters": parameters, "split_mode": v["split_mode"],
        })
    (output_dir / "models.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Catálogo de modelos construido:", output_dir / "models.json")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/web")
    parser.add_argument("--dataset", default="data/demo.txt")
    parser.add_argument("--checkpoint-dir", default="checkpoints/web")
    args = parser.parse_args()
    build(args.output_dir, args.dataset, args.checkpoint_dir)
