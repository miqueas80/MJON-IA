"""MJON Distill Bench: medición del aporte del maestro vs alumno previo.

Reporta NLL sobre conceptos *no enseñados*, recuperación escolar y muestras
de respuestas sin usar reglas preprogramadas para producir la predicción.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from mjon_ia.finetune_v05 import encode_record, collate
from benchmark_v05 import predict, normalize
from benchmark_v07 import balanced, evaluate as school_evaluate, summarize


def instantiate(file):
    obj=torch.load(file,map_location="cpu",weights_only=True)
    codec=ByteBPETokenizer.from_dict(obj["tokenizer"])
    model=ByteTransformer(ModelConfig(**obj["model_config"])).eval()
    model.load_state_dict(obj["state_dict"])
    return model,codec,obj


@torch.inference_mode()
def conditional_loss(model,codec,rows):
    values=[]
    for row in rows:
        try:x,y=encode_record(codec,row["question"],row["answer"],model.config.context)
        except ValueError:continue
        bx,by=collate([(x,y)])
        logits,_=model(bx)
        values.append(F.cross_entropy(
            logits.reshape(-1,logits.shape[-1]),by.reshape(-1),ignore_index=-100).item())
    if not values:raise ValueError("No hay textos evaluables dentro del contexto")
    return round(sum(values)/len(values),5),len(values)


def run(before,after,teacher,school_holdout,school_transfer,out,per_subject=10):
    torch.set_num_threads(2)
    info=json.loads(Path(teacher).read_text(encoding="utf-8"))
    if info.get("schema")!="mjonia.teacher.v1":raise ValueError("Sin procedencia de maestro")
    held=info["holdout"]
    if not held:raise ValueError("Profesor no entregó ejemplos reservados")
    known=balanced(json.loads(Path(school_holdout).read_text(encoding="utf-8")),per_subject)
    transfer=balanced(json.loads(Path(school_transfer).read_text(encoding="utf-8")),per_subject)
    summary=[]
    for checkpoint in (before,after):
        model,codec,obj=instantiate(checkpoint)
        loss,used=conditional_loss(model,codec,held)
        examples=[]
        for row in held[:6]:
            prediction=predict(model,codec,row["question"],max_tokens=36)
            examples.append({"subject":row["subject"],"question":row["question"],
                             "reference_teacher":row["answer"],
                             "generated":prediction[:160],
                             "exact_reference":normalize(prediction)==normalize(row["answer"])})
        school=summarize(school_evaluate(model,codec,known))
        shifted=summarize(school_evaluate(model,codec,transfer))
        record={"model":Path(checkpoint).stem,
                "teacher_holdout_loss":loss,"teacher_holdout_evaluated":used,
                "teacher_exact":sum(x["exact_reference"] for x in examples),
                "teacher_exact_total":len(examples),
                "school_holdout":school,"school_transfer":shifted,
                "examples":examples}
        summary.append(record)
        print("MJON Destilación:",record["model"],
              "profesor NLL",record["teacher_holdout_loss"],
              "escolar",school["correct"],"/",school["total"],
              "transfer",shifted["correct"],"/",shifted["total"],flush=True)
    result={
        "schema":"mjonia.distillation.v1",
        "teacher":info["teacher_id"],"teacher_revision":info["teacher_revision"],
        "teacher_weight_license":info["teacher_weight_license"],
        "training_examples":len(info["training"]),"holdout_examples":len(held),
        "models":summary,
        "limitations":"El profesor también puede equivocarse; semejanza con sus respuestas no implica veracidad ni inteligencia general. "
                     "La comparación escolar usa enunciados independientes, pero las materias aún son ejercicios sintéticos."
    }
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v07.pt")
    p.add_argument("--after",default="checkpoints/web/v08.pt")
    p.add_argument("--teacher",default="data/teacher_v08.json")
    p.add_argument("--school-holdout",default="data/school_v07/holdout.json")
    p.add_argument("--school-transfer",default="data/school_v07/transfer.json")
    p.add_argument("--out",default="docs/web/distillation.json")
    p.add_argument("--per-subject",type=int,default=10)
    a=p.parse_args()
    run(a.before,a.after,a.teacher,a.school_holdout,a.school_transfer,a.out,a.per_subject)
