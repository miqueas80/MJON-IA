"""Publica checkpoint v0.6 sin descartar variantes anteriores."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint, catalog_path, weights_path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("version") != 6:
        raise ValueError("Se requiere checkpoint v0.6")
    catalog_path = Path(catalog_path)
    weights_path = Path(weights_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("format") != "mjonia.models.v1" or any(x["id"] == "v06" for x in catalog["models"]):
        raise ValueError("Catálogo incompatible o v0.6 duplicado")
    storage = {}
    for tensor in state["state_dict"].values():
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.numel())
        storage[key] = tensor.numel()
    record = {
        "id": "v06",
        "label": "v0.6 · memoria contextual",
        "description": "Transformer propio ajustado para recordar atributos de turnos anteriores y aceptar correcciones. No tiene memoria persistente ni capacidades generales.",
        "file": weights_path.name,
        "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "steps": int(state["step"]), "context": int(state["model_config"]["context"]),
        "parameters": sum(storage.values()), "tokenizer": "bpe",
        "training": "sft-memory", "split_mode": "heldout-objects"
    }
    catalog["models"].append(record)
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Modelo v0.6 agregado:", record, flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/web/v06.pt")
    p.add_argument("--catalog", default="docs/web/models.json")
    p.add_argument("--weights", default="docs/web/weights-v06.txt")
    args = p.parse_args()
    run(args.checkpoint, args.catalog, args.weights)
