"""MJON v0.9: continúa entrenando el MISMO Transformer con más ejemplos Qwen.

Protege el mejor checkpoint según validación de conceptos NO entrenados y
pérdida de repaso escolar. Solo son pesos de MJON Original.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .finetune_v05 import collate, encode_record, evaluate
from .finetune_v06 import read_examples
from .model import ByteTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer
from .teacher_v08 import verify_separation
from .train_v03 import safe_save


def load_pools(base, teacher_file, school_file, memory_file):
    base_obj=torch.load(base,map_location="cpu",weights_only=True)
    if base_obj.get("version")!=8:raise ValueError("La base debe ser MJON Original v0.8")
    teacher=json.loads(Path(teacher_file).read_text(encoding="utf-8"))
    if teacher.get("generation_version")!=9 or teacher.get("teacher_weight_license")!="Apache-2.0":
        raise ValueError("Maestro sin procedencia v0.9")
    verify_separation(teacher["training"],teacher["holdout"])
    _,school=read_examples(school_file)
    _,memory=read_examples(memory_file)
    config=ModelConfig(**base_obj["model_config"])
    codec=ByteBPETokenizer.from_dict(base_obj["tokenizer"])
    pools={}
    for key,rows in (("teacher",teacher["training"]),("holdout",teacher["holdout"]),
                     ("school",school),("memory",memory)):
        enc=[]
        for row in rows:
            try:
                enc.append(encode_record(codec,row["question"],row["answer"],config.context))
            except ValueError:
                if key in ("teacher","holdout"):
                    raise  # Los ejemplos docentes se verifican antes: no descartarlos sin avisar.
                continue  # El repaso largo puede exceder el contexto del alumno.
        if not enc:
            raise ValueError(f"No hay ejemplos compatibles: {key}")
        pools[key]=enc
    if len(pools["teacher"])<120 or len(pools["holdout"])<6:
        raise ValueError("Faltan ejemplos docentes o de evaluación")
    return base_obj,teacher,pools


def metrics(model,pools):
    holdout=evaluate(model,pools["holdout"],count=len(pools["holdout"]))
    school=evaluate(model,pools["school"],count=20)
    memory=evaluate(model,pools["memory"],count=20)
    return {"teacher_holdout_loss":holdout,
            "school_replay_loss":school,"memory_replay_loss":memory,
            "selection_score":round(holdout + .15*school + .10*memory,5)}


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError("Parámetros numéricos inválidos")
    if not 0<args.teacher_fraction<1 or not 0<=args.memory_fraction<1-args.teacher_fraction:
        raise ValueError("Fracciones de mezcla inválidas")
    if args.lr<=0:raise ValueError("lr debe ser positiva")
    torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    base,teacher,pools=load_pools(args.base,args.teacher,args.replay,args.memory)
    model=ByteTransformer(ModelConfig(**base["model_config"]))
    model.load_state_dict(base["state_dict"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    inputs={"base_sha256":hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
            "teacher_sha256":hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest(),
            "replay_sha256":hashlib.sha256(Path(args.replay).read_bytes()).hexdigest(),
            "memory_sha256":hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
            "lr":args.lr,"seed":args.seed,"batch_size":args.batch_size,
            "teacher_fraction":args.teacher_fraction,"memory_fraction":args.memory_fraction}
    best_path=Path(args.out)
    latest_path=best_path.with_name(best_path.stem+".last.pt")
    history=[];start=0
    initial=metrics(model,pools)
    best_score=float(initial["selection_score"])
    best_step=0
    if args.resume:
        resume=torch.load(args.resume,map_location="cpu",weights_only=True)
        if resume.get("version")!=9 or any(resume.get(k)!=v for k,v in inputs.items()):
            raise ValueError("Checkpoint incompatible")
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        torch.set_rng_state(resume["torch_rng_state"])
        history=list(resume["history"])
        start=int(resume["step"])
        best_score=float(resume["best_score"]);best_step=int(resume["best_step"])
        if start>=args.steps:raise ValueError("steps debe ser superior al paso guardado")
    else:
        history.append({"step":0,**initial,"train_loss":None})
        print("MJON v0.9 · punto de partida",initial,flush=True)
    def snapshot(step):
        return {"version":9,**inputs,"model_config":base["model_config"],
                "tokenizer":base["tokenizer"],"state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"step":step,
                "teacher_id":teacher["teacher_id"],
                "teacher_revision":teacher["teacher_revision"],
                "best_score":best_score,"best_step":best_step,
                "history":history}
    if not args.resume:
        safe_save(snapshot(0),best_path)
    for step in range(start+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        items=[]
        for _ in range(args.batch_size):
            r=torch.rand(()).item()
            group=("teacher" if r<args.teacher_fraction else
                   "memory" if r<args.teacher_fraction+args.memory_fraction else "school")
            pool=pools[group]
            idx=int(torch.randint(len(pool),(1,)).item())
            items.append(pool[idx])
        tokens,labels=collate(items)
        logits,_=model(tokens)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                             labels.reshape(-1),ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            results=metrics(model,pools)
            row={"step":step,"train_loss":round(float(loss.item()),5),**results}
            history.append(row)
            score=float(results["selection_score"])
            if score<best_score-0.00001:
                best_score=score;best_step=step
                safe_save(snapshot(step),best_path)
            print(f"MJON v0.9 paso {step}; mejor {best_step}:",row,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(snapshot(step),latest_path)
    best=torch.load(best_path,map_location="cpu",weights_only=True)
    print(f"MJON v0.9: mejor checkpoint paso {best['step']} / {args.steps}; "
          f"score inicial {initial['selection_score']}, mejor {best['best_score']}.",flush=True)
    best_path.with_suffix(".metrics.json").write_text(
        json.dumps({"initial":initial,"best_step":best["step"],
                    "best_score":best["best_score"],"history":history},
                   indent=2)+"\n",encoding="utf-8")
    return best_path


def parse_args():
    p=argparse.ArgumentParser(description="Transferencia real v0.9 al Transformer MJON")
    p.add_argument("--base",default="checkpoints/web/v08.pt")
    p.add_argument("--teacher",default="data/teacher_v09.json")
    p.add_argument("--replay",default="data/school_v07/train.json")
    p.add_argument("--memory",default="data/memory_v06/train.json")
    p.add_argument("--out",default="checkpoints/web/v09.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=700)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--teacher-fraction",type=float,default=.65)
    p.add_argument("--memory-fraction",type=float,default=.12)
    p.add_argument("--lr",type=float,default=.00008)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--seed",type=int,default=909)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
