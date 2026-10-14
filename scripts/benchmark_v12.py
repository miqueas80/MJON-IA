"""Prueba independiente: ¿las relaciones Graphify llegaron a las neuronas MJON?

Se generan respuestas del Transformer MJON, sin consultar Graphify al inferir.
El examen usa aristas y formulaciones NO usadas para seleccionar checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
from benchmark_v07 import load,balanced,evaluate as school_evaluate,summarize
from benchmark_v05 import normalize,predict
from benchmark_v06 import select_balanced


def score(model,codec,rows):
    output=[]
    for record in rows:
        actual=predict(model,codec,record["question"],max_tokens=8)
        output.append({"source":record["source"],"target":record["target"],
            "relation":record["relation"],"question":record["question"],
            "expected":record["answer"],"generated":actual[:80],
            "correct":normalize(actual)==normalize(record["answer"]),
            "edge_status":record["edge_status"]})
    by_label={}
    for label in ("EXTRACTED","NOT_RECORDED"):
        subset=[r for r in output if r["edge_status"]==label]
        by_label[label]={"correct":sum(r["correct"] for r in subset),"total":len(subset)}
    return {"correct":sum(r["correct"] for r in output),"total":len(output),
            "by_label":by_label,"answers":output}


def run(before,after,graph_test,school,transfer,memory,out):
    torch.set_num_threads(2)
    graph=json.loads(Path(graph_test).read_text(encoding="utf-8"))
    if len(graph)<4:raise ValueError("El examen de Graphify está vacío")
    school_rows=balanced(json.loads(Path(school).read_text(encoding="utf-8")),10)
    transfer_rows=balanced(json.loads(Path(transfer).read_text(encoding="utf-8")),10)
    memory_rows=select_balanced(json.loads(Path(memory).read_text(encoding="utf-8")),4)
    results=[]
    for file in (before,after):
        model,codec=load(file)
        relations=score(model,codec,graph)
        school_results=summarize(school_evaluate(model,codec,school_rows))
        transfer_results=summarize(school_evaluate(model,codec,transfer_rows))
        correct=0
        for row in memory_rows:
            actual=predict(model,codec,row["question"],max_tokens=16)
            correct+=normalize(actual)==normalize(row["answer"])
        rec={"model":Path(file).stem,"graph_test":relations,
             "school":school_results,"transfer":transfer_results,
             "memory":{"correct":correct,"total":len(memory_rows)}}
        results.append(rec)
        print(f"MJON Graphify examen {rec['model']}: "
              f"grafo {relations['correct']}/{relations['total']} "
              f"escuela {school_results['correct']}/40 "
              f"reformulación {transfer_results['correct']}/40 "
              f"memoria {correct}/{len(memory_rows)}",flush=True)
    ck=torch.load(after,map_location="cpu",weights_only=True)
    report={"schema":"mjonia.graphify.eval.v1","source":"Graphify-Labs/graphify",
            "confidence":"EXTRACTED only","models":results,
            "selected_step":int(ck["step"]),
            "baseline_selected":int(ck["step"])==0,
            "notes":"Aristas de código, no conocimiento general. Los negativos son aristas no registradas "
                    "por Graphify, NO afirmaciones necesariamente falsas sobre el software. "
                    "Los pesos del alumno se entrenaron mediante gradientes, no hay consulta de grafo en la app."}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v11.pt")
    p.add_argument("--after",default="checkpoints/web/v12.pt")
    p.add_argument("--graph-test",default="data/graphify_v12/test.json")
    p.add_argument("--school",default="data/school_v07/holdout.json")
    p.add_argument("--transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="docs/web/graphify-v12.json")
    a=p.parse_args()
    run(a.before,a.after,a.graph_test,a.school,a.transfer,a.memory,a.out)
