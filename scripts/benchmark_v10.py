"""Examen v0.10: conceptos del maestro fuera de calibración y pruebas previas.

Las últimas preguntas del holdout Qwen están completamente reservadas: nunca
se usan para gradientes ni para seleccionar checkpoints.
"""
import argparse
import json
from pathlib import Path

import torch

from benchmark_v07 import balanced, load, evaluate, summarize
from benchmark_v05 import normalize, predict
from benchmark_v06 import select_balanced


def run(before,after,teacher,school,transfer,memory,output):
    torch.set_num_threads(2)
    doc=json.loads(Path(teacher).read_text(encoding="utf-8"))
    unseen=doc["holdout"]
    unseen=unseen[max(1,len(unseen)//2):]
    if len(unseen)<3:raise ValueError("Faltan conceptos reservados para el examen")
    school_cases=balanced(json.loads(Path(school).read_text(encoding="utf-8")),10)
    transfer_cases=balanced(json.loads(Path(transfer).read_text(encoding="utf-8")),10)
    memory_cases=select_balanced(json.loads(Path(memory).read_text(encoding="utf-8")),4)
    scores=[]
    for path in (before,after):
        model,codec=load(path)
        master=[]
        for row in unseen:
            generated=predict(model,codec,row["question"],max_tokens=36)
            master.append({"subject":row["subject"],"question":row["question"],
                           "reference":row["answer"],"generated":generated,
                           "exact":normalize(generated)==normalize(row["answer"])})
        mat=summarize(evaluate(model,codec,school_cases))
        reform=summarize(evaluate(model,codec,transfer_cases))
        memory_results=[]
        for row in memory_cases:
            output_text=predict(model,codec,row["question"],max_tokens=16)
            memory_results.append({"case":row["case"],"question":row["question"],
                    "reference":row["answer"],"generated":output_text,
                    "exact":normalize(output_text)==normalize(row["answer"])})
        result={"model":Path(path).stem,
                "teacher_unseen_exact":{"correct":sum(x["exact"] for x in master),
                                        "total":len(master)},
                "school":mat,"transfer":reform,
                "memory":{"correct":sum(x["exact"] for x in memory_results),
                          "total":len(memory_results)},
                "answers":{"teacher":master,"memory":memory_results}}
        scores.append(result)
        print(f"MJON examen {result['model']}: docente {result['teacher_unseen_exact']['correct']}/{len(master)}, "
              f"escolar {mat['correct']}/{mat['total']}, reformuladas {reform['correct']}/{reform['total']}, "
              f"memoria {result['memory']['correct']}/{len(memory_results)}",flush=True)
    checkpoint=torch.load(after,map_location="cpu",weights_only=True)
    report={"schema":"mjonia.continual.v1","models":scores,
            "selected_step":int(checkpoint["step"]),
            "baseline_selected":int(checkpoint["step"])==0,
            "teacher":"Qwen/Qwen2.5-0.5B-Instruct usado solo en entrenamiento",
            "notes":"Examen final disjunto de la calibración de checkpoints. "
                    "La precisión literal de respuestas del profesor no equivale a comprensión general."}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v08.pt")
    p.add_argument("--after",default="checkpoints/web/v10.pt")
    p.add_argument("--teacher",default="data/teacher_v09.json")
    p.add_argument("--school",default="data/school_v07/holdout.json")
    p.add_argument("--transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="docs/web/continual-v10.json")
    a=p.parse_args()
    run(a.before,a.after,a.teacher,a.school,a.transfer,a.memory,a.out)
