"""Registra MJON v1.2: misma red neuronal entrenada con relaciones de Graphify."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def run(checkpoint,catalog,weights):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if ck.get("version")!=12 or ck.get("graphify_source")!="Graphify-Labs/graphify":
        raise ValueError("El checkpoint no es MJON Graphify v1.2")
    info=json.loads(Path(catalog).read_text(encoding="utf-8"))
    if info.get("format")!="mjonia.models.v1" or any(x["id"]=="v12" for x in info["models"]):
        raise ValueError("Catálogo inválido")
    seen={}
    for t in ck["state_dict"].values():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),t.numel())
        seen[key]=t.numel()
    info["models"].append({
        "id":"v12","label":"v1.2 · neuronas con relaciones Graphify",
        "description":"Transformer original MJON actualizado mediante ejemplos de relaciones EXTRACTED del grafo Graphify de su código. Graphify solo opera durante el entrenamiento: no hay grafo ni otro motor en la app. Un paso 0 indica que no hubo mejora aceptada.",
        "file":Path(weights).name,
        "sha256":hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "steps":int(ck["step"]),"context":ck["model_config"]["context"],
        "parameters":sum(seen.values()),"tokenizer":"bpe",
        "training":"graphify-extracted-graph-SFT+anti-forgetting",
        "split_mode":"heldout-edges-and-paraphrases",
        "graphify_license":ck["graphify_license"],"graphify_source":ck["graphify_source"],
        "graphify_graph_sha256":ck["graphify_graph_sha256"],
        "selected_step":int(ck["best_step"]),"baseline_selected":int(ck["step"])==0
    })
    Path(catalog).write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MJON Graphify catálogo v1.2:",ck["step"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/web/v12.pt")
    p.add_argument("--catalog",default="docs/web/models.json")
    p.add_argument("--weights",default="docs/web/weights-v12.txt")
    a=p.parse_args()
    run(a.checkpoint,a.catalog,a.weights)
