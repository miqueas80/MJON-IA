"""MJON Examen: evaluación publicada de cuatro materias y reformulación.

Las respuestas las genera el Transformer de forma greedy, nunca un solucionador
de reglas. Baseline v0.6 vs v0.7; cada fallo queda en informe reproducible.
"""
import argparse
import json
from pathlib import Path

import torch
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from benchmark_v05 import normalize, predict
from benchmark_v06 import select_balanced


def load(checkpoint):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    codec=ByteBPETokenizer.from_dict(state["tokenizer"])
    model=ByteTransformer(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["state_dict"])
    return model,codec


def balanced(rows,per_subject=10):
    selected=[]
    for subject in ("matematica","lengua","ciencias","logica"):
        chosen=[r for r in rows if r["subject"]==subject][:per_subject]
        if len(chosen)!=per_subject:raise ValueError("No hay suficientes preguntas reservadas")
        selected.extend(chosen)
    return selected


def evaluate(model,codec,rows):
    results=[]
    for row in rows:
        response=predict(model,codec,row["question"],max_tokens=16)
        results.append({"subject":row.get("subject"),"mode":row.get("mode"),
                        "question":row["question"],"expected":row["answer"],
                        "generated":response[:100],"exact":normalize(response)==normalize(row["answer"])})
    return results


def summarize(rows):
    subjects={}
    for subject in ("matematica","lengua","ciencias","logica"):
        subset=[r for r in rows if r["subject"]==subject]
        subjects[subject]={"correct":sum(r["exact"] for r in subset),"total":len(subset)}
    return {"correct":sum(r["exact"] for r in rows),"total":len(rows),"subjects":subjects}


def run(before,after,holdout,transfer,memory_holdout,firstwords,out,per_subject=10):
    torch.set_num_threads(2)
    known=balanced(json.loads(Path(holdout).read_text(encoding="utf-8")),per_subject)
    shifted=balanced(json.loads(Path(transfer).read_text(encoding="utf-8")),per_subject)
    raw_memory=json.loads(Path(memory_holdout).read_text(encoding="utf-8"))
    mem_recall=select_balanced(raw_memory,per_group=4)
    raw_firstwords=json.loads(Path(firstwords).read_text(encoding="utf-8"))[:32]
    models=[]
    for checkpoint in (before,after):
        model,codec=load(checkpoint)
        unseen=evaluate(model,codec,known)
        reworded=evaluate(model,codec,shifted)
        memory=[]
        for row in mem_recall:
            actual=predict(model,codec,row["question"],max_tokens=16)
            memory.append({"exact":normalize(actual)==normalize(row["answer"]),
                           "expected":row["answer"],"generated":actual[:80],
                           "case":row["case"]})
        basic=[]
        for row in raw_firstwords:
            actual=predict(model,codec,row["question"],max_tokens=16)
            basic.append(normalize(actual)==normalize(row["answer"]))
        models.append({"checkpoint":Path(checkpoint).stem,
            "holdout":summarize(unseen),"transfer":summarize(reworded),
            "previous_memory":{"correct":sum(x["exact"] for x in memory),"total":len(memory)},
            "previous_short":{"correct":sum(basic),"total":len(basic)},
            "answers":{"holdout":unseen,"transfer":reworded,"memory":memory}})
        print("MJON Examen:",models[-1]["checkpoint"],
              "holdout",models[-1]["holdout"]["correct"],"/",len(unseen),
              "transfer",models[-1]["transfer"]["correct"],"/",len(reworded),
              "memory",sum(x["exact"] for x in memory),"/",len(memory),
              "short",sum(basic),"/",len(basic),flush=True)
    report={"schema":"mjonia.school.v1","models":models,
            "limitations":"Ejercicios sintéticos de cuatro materias, no equivalen a secundaria real. "
            "Transfer usa redacciones nuevas y en Matemática números fuera del rango de entrenamiento. "
            "Las soluciones de referencia son reglas del dataset, pero el modelo responde sin un solucionador externo."}
    destination=Path(out);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v06.pt")
    p.add_argument("--after",default="checkpoints/web/v07.pt")
    p.add_argument("--holdout",default="data/school_v07/holdout.json")
    p.add_argument("--transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory-holdout",default="data/memory_v06/holdout.json")
    p.add_argument("--firstwords",default="data/instruction_v05/holdout.json")
    p.add_argument("--out",default="docs/web/school.json")
    p.add_argument("--per-subject",type=int,default=10)
    a=p.parse_args()
    run(a.before,a.after,a.holdout,a.transfer,a.memory_holdout,a.firstwords,a.out,a.per_subject)
