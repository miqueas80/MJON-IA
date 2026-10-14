"""Registra v0.7 en el selector sin sustituir los modelos anteriores."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint,catalog_path,weights_path):
    checkpoint=Path(checkpoint);catalog_path=Path(catalog_path);weights_path=Path(weights_path)
    obj=torch.load(checkpoint,weights_only=True,map_location="cpu")
    if obj.get("version")!=7:raise ValueError("Checkpoint debe ser v0.7")
    data=json.loads(catalog_path.read_text(encoding="utf-8"))
    if data.get("format")!="mjonia.models.v1" or any(x.get("id")=="v07" for x in data["models"]):
        raise ValueError("Catálogo incompatible o duplicado")
    unique={}
    for tensor in obj["state_dict"].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        unique[key]=tensor.numel()
    entry={"id":"v07","label":"v0.7 · MJON Escuela",
           "description":"Cuatro materias sintéticas, BPE propio, 256 tokens de contexto y SFT multiasignatura. No equivale a aprobar secundaria.",
           "file":weights_path.name,
           "sha256":hashlib.sha256(weights_path.read_bytes()).hexdigest(),
           "steps":obj["step"],"context":obj["model_config"]["context"],
           "parameters":sum(unique.values()),"tokenizer":"bpe",
           "training":"sft-school","split_mode":"heldout-and-rephrased"}
    data["models"].append(entry)
    catalog_path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Publicado MJON Escuela:",entry,flush=True)
    return entry

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v07.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v07.txt")
    a=p.parse_args();run(a.checkpoint,a.catalog,a.weights)
