"""MJON v1.1: examen con conceptos enseñados reformulados y conceptos nuevos.

Se evalúa generación real de MJON, NO recuperación de respuestas de un índice.
Dev: 30 paráfrasis. Final: otras 70 paráfrasis + 36 conceptos desconocidos.
"""
import argparse
import json
from pathlib import Path

import torch

from benchmark_v07 import load, balanced, evaluate as school_evaluate, summarize
from benchmark_v05 import normalize, predict
from benchmark_v06 import select_balanced


def report_cases(model,codec,rows,max_tokens=35):
    scored=[]
    for r in rows:
        generated=predict(model,codec,r["question"],max_tokens=max_tokens)
        scored.append({
            "subject":r.get("subject"),"concept":r.get("concept"),
            "question":r["question"],"reference":r["answer"],
            "generated":generated[:160],
            "exact":normalize(generated)==normalize(r["answer"])
        })
    groups={}
    for area in sorted(set(r.get("subject") for r in rows)):
        subset=[r for r in scored if r["subject"]==area]
        groups[area]={"correct":sum(r["exact"] for r in subset),"total":len(subset)}
    return {"correct":sum(r["exact"] for r in scored),
            "total":len(scored),"by_subject":groups,"examples":scored}


def run(before,after,paraphrase,unknown,school,transfer,memory,out):
    torch.set_num_threads(2)
    taught=json.loads(Path(paraphrase).read_text(encoding="utf-8"))
    new=json.loads(Path(unknown).read_text(encoding="utf-8"))
    if len(taught)<100 or len(new)<40:raise ValueError("Examen curricular incompleto")
    taught_train_concepts={r["concept"] for r in taught}
    if taught_train_concepts & {r["concept"] for r in new}:
        raise ValueError("Conceptos reservados se superponen")
    # La calibración del entrenador usa las primeras 30 paráfrasis.
    fresh=taught[30:]
    school_rows=balanced(json.loads(Path(school).read_text(encoding="utf-8")),10)
    transfer_rows=balanced(json.loads(Path(transfer).read_text(encoding="utf-8")),10)
    memory_rows=select_balanced(json.loads(Path(memory).read_text(encoding="utf-8")),4)
    scores=[]
    for checkpoint in (before,after):
        model,codec=load(checkpoint)
        def eval_memory():
            rows=[]
            for rec in memory_rows:
                prediction=predict(model,codec,rec["question"],max_tokens=16)
                rows.append({"case":rec["case"],"question":rec["question"],
                            "generated":prediction,"reference":rec["answer"],
                            "exact":normalize(prediction)==normalize(rec["answer"])})
            return {"correct":sum(r["exact"] for r in rows),
                    "total":len(rows),"samples":rows}
        result={
            "model":Path(checkpoint).stem,
            "taught_paraphrases":report_cases(model,codec,fresh),
            "unseen_concepts":report_cases(model,codec,new),
            "school":summarize(school_evaluate(model,codec,school_rows)),
            "transfer":summarize(school_evaluate(model,codec,transfer_rows)),
            "memory":eval_memory()
        }
        print(f"MJON Conocimiento {result['model']}: "
              f"paráfrasis {result['taught_paraphrases']['correct']}/{len(fresh)}, "
              f"conceptos nuevos {result['unseen_concepts']['correct']}/{len(new)}, "
              f"escuela {result['school']['correct']}/40, "
              f"transferencia {result['transfer']['correct']}/40, "
              f"memoria {result['memory']['correct']}/{len(memory_rows)}",flush=True)
        scores.append(result)
    ck=torch.load(after,map_location="cpu",weights_only=True)
    output={
        "schema":"mjonia.knowledge.v1","baseline_model":Path(before).stem,
        "selected_model":Path(after).stem,"models":scores,
        "selected_step":int(ck["step"]),"baseline_selected":int(ck["step"])==0,
        "scope":"Examen de 70 paráfrasis no usadas para seleccionar checkpoint, "
                "40 consultas sobre conceptos jamás enseñados, 40 preguntas escolares, "
                "40 reformuladas y memoria. Exactitud literal de texto; "
                "no demuestra comprensión general ni iguala ChatGPT."
    }
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v10.pt")
    p.add_argument("--after",default="checkpoints/web/v11.pt")
    p.add_argument("--paraphrase",default="data/knowledge_v11/paraphrase.json")
    p.add_argument("--unknown",default="data/knowledge_v11/unknown.json")
    p.add_argument("--school",default="data/school_v07/holdout.json")
    p.add_argument("--transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="docs/web/knowledge-v11.json")
    args=p.parse_args()
    run(args.before,args.after,args.paraphrase,args.unknown,args.school,
        args.transfer,args.memory,args.out)
