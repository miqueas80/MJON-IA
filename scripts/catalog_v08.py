"""Registra MJON IA v0.8: alumno destilado desde un modelo abierto."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint,catalog_path,weights_path):
    obj=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if obj.get("version")!=8 or obj.get("teacher_id")!="Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("Checkpoint no es v0.8 con profesor declarado")
    path=Path(catalog_path);weights=Path(weights_path)
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("format")!="mjonia.models.v1" or any(x["id"]=="v08" for x in data["models"]):
        raise ValueError("Catálogo incompatible o v0.8 duplicado")
    unique={}
    for t in obj["state_dict"].values():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),t.numel())
        unique[key]=t.numel()
    data["models"].append({
        "id":"v08","label":"v0.8 · alumno del modelo abierto",
        "description":"Transformer MJON propio ajustado con respuestas de Qwen2.5-0.5B-Instruct (Apache-2.0); NO contiene los pesos de Qwen. Capacidad experimental, verificar resultados.",
        "file":weights.name,"sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
        "steps":obj["step"],"context":obj["model_config"]["context"],
        "parameters":sum(unique.values()),"tokenizer":"bpe","training":"teacher-distillation",
        "teacher_id":obj["teacher_id"],"teacher_revision":obj["teacher_revision"],
        "split_mode":"heldout-topics"
    })
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("MJON v0.8 registrado con licencia/procedencia:",obj["teacher_id"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v08.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v08.txt")
    a=p.parse_args()
    run(a.checkpoint,a.catalog,a.weights)
