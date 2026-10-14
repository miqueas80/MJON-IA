"""Registra la versión MJON entrenada con enseñanza adicional del maestro abierto."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint="checkpoints/web/v09.pt",catalog="docs/web/models.json",
        weights="docs/web/weights-v09.txt"):
    trained=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if trained.get("version")!=9 or trained.get("teacher_id")!="Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("El checkpoint no es MJON Original v0.9 destilado")
    catalog_path=Path(catalog)
    output=Path(weights)
    listing=json.loads(catalog_path.read_text(encoding="utf-8"))
    if listing.get("format")!="mjonia.models.v1" or any(v["id"]=="v09" for v in listing["models"]):
        raise ValueError("Catálogo inválido o v09 ya registrada")
    storages={}
    for tensor in trained["state_dict"].values():
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        storages[key]=tensor.numel()
    listing["models"].append({
        "id":"v09","label":"v0.9 · MJON Original con entrenamiento ampliado",
        "description":"El mismo Transformer propio entrenado desde cero, con más ejemplos de Qwen durante entrenamiento; el modelo publicado NO usa Qwen, ni contiene sus pesos. Mejora no garantizada.",
        "file":output.name,
        "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "steps":int(trained["step"]),"context":trained["model_config"]["context"],
        "parameters":sum(storages.values()),"tokenizer":"bpe",
        "training":"expanded-response-distillation","teacher_id":trained["teacher_id"],
        "teacher_revision":trained["teacher_revision"],
        "split_mode":"heldout-concepts","best_step":int(trained["best_step"])
    })
    catalog_path.write_text(json.dumps(listing,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Registrado el mejor checkpoint MJON Original:",trained["best_step"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v09.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v09.txt")
    a=p.parse_args()
    run(a.checkpoint,a.catalog,a.weights)
