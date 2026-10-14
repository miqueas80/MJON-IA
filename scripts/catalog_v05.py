"""Catálogo público del modelo SFT v0.5 con pesos propios."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def run(checkpoint="checkpoints/web/v05.pt",catalog="docs/web/models.json",weights="docs/web/weights-v05.txt"):
    obj=torch.load(checkpoint,weights_only=True,map_location="cpu")
    if obj.get("version")!=5:raise ValueError("Checkpoint no es SFT v0.5")
    path=Path(catalog);weights_path=Path(weights)
    info=json.loads(path.read_text(encoding="utf-8"))
    if info.get("format")!="mjonia.models.v1" or any(x["id"]=="v05" for x in info["models"]):
        raise ValueError("Catálogo inválido o modelo duplicado")
    storages={}
    for tensor in obj["state_dict"].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        storages[key]=tensor.numel()
    result={
        "id":"v05","label":"v0.5 · primeras palabras (SFT)",
        "description":"Modelo propio ajustado desde v0.4 para extraer colores, materiales y estados en respuestas cortas. No es un asistente general.",
        "file":weights_path.name,"sha256":hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "steps":int(obj["step"]),"context":int(obj["model_config"]["context"]),
        "parameters":sum(storages.values()),"tokenizer":"bpe","training":"sft",
        "split_mode":"heldout-objects"
    }
    info["models"].append(result)
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Modelo v0.5 agregado:",result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v05.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v05.txt")
    a=p.parse_args();run(a.checkpoint,a.catalog,a.weights)
