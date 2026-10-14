"""Añade MJON v0.10, siempre un único Transformer original."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint,catalog,weights):
    obj=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if obj.get("version")!=10 or obj.get("teacher_id")!="Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("Checkpoint MJON v0.10 incorrecto")
    doc=json.loads(Path(catalog).read_text(encoding="utf-8"))
    if doc.get("format")!="mjonia.models.v1" or any(x.get("id")=="v10" for x in doc["models"]):
        raise ValueError("Catálogo inválido")
    unique={}
    for tensor in obj["state_dict"].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        unique[key]=tensor.numel()
    doc["models"].append({
        "id":"v10","label":"v1.0 · MJON Original antiolvido",
        "description":"El mismo Transformer MJON (~1,29M) con entrenamiento de conocimiento del maestro Qwen, repaso y destilación funcional de habilidades previas. Puede conservar el checkpoint inicial si ningún candidato supera el control de regresión. Sin motor externo.",
        "file":Path(weights).name,
        "sha256":hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "steps":int(obj["step"]),"context":obj["model_config"]["context"],
        "parameters":sum(unique.values()),"tokenizer":"bpe",
        "training":"continual-distillation-replay","split_mode":"separate-dev-and-final",
        "teacher_id":obj["teacher_id"],"teacher_revision":obj["teacher_revision"],
        "selected_step":int(obj["best_step"]),"baseline_selected":int(obj["step"])==0
    })
    Path(catalog).write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Catálogo MJON v1.0: checkpoint",obj["step"],
          "(0 significa que se rechazaron los ajustes por regresión)",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v10.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v10.txt")
    a=p.parse_args()
    run(a.checkpoint,a.catalog,a.weights)
