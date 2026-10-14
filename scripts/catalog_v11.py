"""Registra el checkpoint MJON Original v1.1, sin pesos ni motores externos."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def run(checkpoint, catalog, weights):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if ck.get("version")!=11 or ck.get("curriculum_source")!="mjon-author-written-cc0":
        raise ValueError("Checkpoint no pertenece al currículo propio")
    meta=json.loads(Path(catalog).read_text(encoding="utf-8"))
    if meta.get("format")!="mjonia.models.v1" or any(x.get("id")=="v11" for x in meta["models"]):
        raise ValueError("Catálogo incompatible o versión repetida")
    unique={}
    for t in ck["state_dict"].values():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),t.numel())
        unique[key]=t.numel()
    meta["models"].append({
        "id":"v11","label":"v1.1 · conocimiento fundamental",
        "description":"Transformer MJON Original entrenado con lecciones escritas para MJON, repaso y protección antiolvido. Ningún modelo externo funciona en la aplicación. Puede conservar la base si falla la selección.",
        "file":Path(weights).name,
        "sha256":hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "steps":int(ck["step"]),"context":int(ck["model_config"]["context"]),
        "parameters":sum(unique.values()),"tokenizer":"bpe",
        "training":"human-authored-curriculum+replay+functional-KD",
        "split_mode":"heldout-phrases-and-concepts",
        "selected_step":int(ck["best_step"]),
        "baseline_selected":int(ck["step"])==0
    })
    Path(catalog).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MJON v1.1 registrado: checkpoint",ck["step"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v11.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v11.txt")
    a=p.parse_args()
    run(a.checkpoint,a.catalog,a.weights)
