"""Examen de generación real de MJON Neural antes y después de SFT.

Respuestas producidas únicamente por los pesos de MJON, no por búsqueda
de coincidencias ni por otro modelo. Conceptos separados por group_id.
"""
import argparse
import json
from pathlib import Path

import torch

from mjon_ia.neural_core import MJONNeuralCore,NeuralConfig
from mjon_ia.neural_sft import read_instructions


def normalize(s):
    return " ".join(s.strip().lower().rstrip(".?!¡¿ \n").split())


@torch.no_grad()
def run(before,after,tokenizer_path,dataset,report,max_cases=40,seed=2026):
    import sentencepiece as spm
    torch.set_num_threads(2)
    tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    _,heldout,provenance=read_instructions(dataset,seed)
    samples=heldout[:max_cases]
    if not samples:raise ValueError("Examen sin casos reservados")
    results=[]
    for ckfile in (before,after):
        weights=torch.load(ckfile,map_location="cpu",weights_only=True)
        if weights.get("format") not in ("mjon-neural-v2","mjon-neural-sft-v1"):
            raise ValueError("Modelo Neural Core inválido")
        config=NeuralConfig(**weights["config"])
        if config.vocab_size!=tokenizer.vocab_size():
            raise ValueError("Tokenizer del examen incompatible")
        model=MJONNeuralCore(config).eval()
        model.load_state_dict(weights["state_dict"])
        scored=[]
        for row in samples:
            prompt=[tokenizer.bos_id()]+tokenizer.encode(
                "Usuario: "+row["instruction"]+"\nAsistente:",out_type=int)
            budget=min(64,config.context-len(prompt))
            if budget<=0:raise ValueError("Contexto insuficiente para examen")
            starting=torch.tensor([prompt],dtype=torch.long)
            ids=model.generate(starting,max_new_tokens=budget,
                               eos_id=tokenizer.eos_id(),temperature=0.)
            new_tokens=ids[0,len(prompt):].tolist()
            if tokenizer.eos_id() in new_tokens:
                new_tokens=new_tokens[:new_tokens.index(tokenizer.eos_id())]
            actual=tokenizer.decode(new_tokens).strip().split("\n",1)[0]
            expected=row["response"]
            scored.append({
                "subject":row.get("subject"),"group":row["group_id"],
                "instruction":row["instruction"],"reference":expected,
                "response":actual[:600],
                "exact":normalize(actual)==normalize(expected),
            })
        results.append({
            "model":Path(ckfile).stem,
            "exact":sum(r["exact"] for r in scored),
            "total":len(scored),"answers":scored,
        })
        print("MJON Neural examen:",results[-1]["model"],
              results[-1]["exact"],"/",len(scored),flush=True)
    output={
        "schema":"mjon-neural-evaluation.v1",
        "split":"holdout groups never used for training, selected deterministically",
        "provenance":provenance,
        "models":results,
        "warning":"Exactitud literal de respuestas; no equivale a comprensión general ni a paridad con otros modelos.",
    }
    destination=Path(report);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",
                           encoding="utf-8")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",required=True)
    p.add_argument("--after",required=True)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--out",default="checkpoints/neural/sft-evaluation.json")
    p.add_argument("--limit",type=int,default=40)
    p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args()
    run(a.before,a.after,a.tokenizer,a.data,a.out,a.limit,a.seed)
