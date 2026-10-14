"""MJON Original v1.1: aprendizaje de conocimiento propio con preservación estricta.

Continúa los pesos del Transformer MJON v1.0. No descarga ni ejecuta un
segundo modelo en entrenamiento ni inferencia: las etiquetas curriculares
son redactadas y versionadas dentro del repositorio.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .distill_v10 import (choose_candidate, evaluate_development,
                         frozen_distribution_loss)
from .finetune_v05 import collate, encode_record, evaluate
from .finetune_v06 import read_examples
from .model import ByteTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save


def load_training(args):
    base=torch.load(args.base,map_location="cpu",weights_only=True)
    if base.get("version")!=10:raise ValueError("Se exige MJON Original v1.0 como base")
    codec=ByteBPETokenizer.from_dict(base["tokenizer"])
    cfg=ModelConfig(**base["model_config"])
    records={}
    for name,path in (("knowledge",args.knowledge),("paraphrase",args.paraphrase),
                      ("unknown",args.unknown),("school",args.school),("memory",args.memory),
                      ("school_holdout",args.school_holdout),
                      ("school_transfer",args.school_transfer),
                      ("memory_holdout",args.memory_holdout)):
        _,records[name]=read_examples(path)
    trained_concepts={x["concept"] for x in records["knowledge"]}
    novel_concepts={x["concept"] for x in records["unknown"]}
    if trained_concepts & novel_concepts:
        raise ValueError("Fuga de conceptos nuevos a entrenamiento")
    known_questions={r["question"] for r in records["knowledge"]+records["school"]+records["memory"]}
    if known_questions & {r["question"] for r in records["paraphrase"]+records["unknown"]}:
        raise ValueError("Fuga de preguntas de conocimiento a entrenamiento")
    if {r["concept"] for r in records["paraphrase"]} != trained_concepts:
        raise ValueError("Paráfrasis no cubren exactamente los temas enseñados")
    if len(records["knowledge"])<400 or len(records["paraphrase"])<100 or len(records["unknown"])<40:
        raise ValueError("Corpus curricular incompleto")
    pools={}
    for name in ("knowledge","paraphrase","school","memory"):
        encoded=[]
        for row in records[name]:
            try:encoded.append(encode_record(codec,row["question"],row["answer"],cfg.context))
            except ValueError:
                if name in ("knowledge","paraphrase"):raise
        if not encoded:raise ValueError("Corpus vacío: "+name)
        pools[name]=encoded
    # Solo 30/100 paráfrasis se usan para selección. Las otras 70 son examen final.
    pools["knowledge_dev"]=[pools["paraphrase"][i] for i in range(30)]
    final_knowledge=records["paraphrase"][30:]
    # La calibración antigua es distinta del benchmark oficial (0:10).
    development=[]
    for subject in ("matematica","lengua","ciencias","logica"):
        development.extend([r for r in records["school_holdout"] if r["subject"]==subject][10:14])
    for subject in ("matematica","lengua","ciencias","logica"):
        development.extend([r for r in records["school_transfer"] if r["subject"]==subject][10:14])
    development.extend([r for r in records["memory_holdout"] if r.get("case") in
                        ("recall","correction")][:8])
    if len(development)!=40:raise ValueError("No hay 40 casos de desarrollo")
    if any(r["question"] in known_questions for r in development):
        raise ValueError("Contaminación en calibración de retención")
    return base,codec,pools,development,final_knowledge


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError("Pasos y lotes deben ser positivos")
    if not 0<args.knowledge_fraction<1 or not 0<args.memory_fraction<1-args.knowledge_fraction:
        raise ValueError("Mezcla curricular y repaso inválida")
    if not 0<=args.kd_weight<=10 or args.lr<=0:
        raise ValueError("Pesos KD o tasa de aprendizaje inválidos")
    torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    base,codec,pools,dev,_=load_training(args)
    cfg=ModelConfig(**base["model_config"])
    student=ByteTransformer(cfg);student.load_state_dict(base["state_dict"])
    reference=ByteTransformer(cfg).eval()
    reference.load_state_dict(base["state_dict"])
    for parameter in reference.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.01)
    signature={
        "base_sha256":hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "knowledge_sha256":hashlib.sha256(Path(args.knowledge).read_bytes()).hexdigest(),
        "paraphrase_sha256":hashlib.sha256(Path(args.paraphrase).read_bytes()).hexdigest(),
        "school_sha256":hashlib.sha256(Path(args.school).read_bytes()).hexdigest(),
        "memory_sha256":hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
        "holdouts_sha256":[hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in (args.unknown,args.school_holdout,args.school_transfer,args.memory_holdout)],
        "seed":args.seed,"lr":args.lr,"batch_size":args.batch_size,
        "knowledge_fraction":args.knowledge_fraction,"memory_fraction":args.memory_fraction,
        "kd_weight":args.kd_weight,"eval_every":args.eval_every
    }
    baseline_dev=evaluate_development(reference,codec,dev)
    baseline_loss=evaluate(reference,pools["knowledge_dev"],count=len(pools["knowledge_dev"]))
    best_loss=baseline_loss
    best_step=0
    start=0
    history=[]
    out=Path(args.out)
    latest=out.with_name(out.stem+".last.pt")
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=True)
        if state.get("version")!=11 or any(state.get(k)!=v for k,v in signature.items()):
            raise ValueError("Checkpoint y corpus incompatibles")
        if state["model_config"]!=base["model_config"] or state["tokenizer"]!=base["tokenizer"]:
            raise ValueError("Configuración/tokenizador incompatibles")
        student.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        start=int(state["step"]);history=list(state["history"])
        best_loss=float(state["best_knowledge_loss"]);best_step=int(state["best_step"])
        if start>=args.steps:raise ValueError("steps debe superar el paso reanudado")
    else:
        history=[{"step":0,"knowledge_dev_loss":baseline_loss,
                  "development":baseline_dev,"promoted":True}]
    def snapshot(step):
        return {"version":11,**signature,
                "model_config":base["model_config"],"tokenizer":base["tokenizer"],
                "state_dict":student.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"step":step,
                "best_step":best_step,"best_knowledge_loss":best_loss,
                "baseline_knowledge_loss":baseline_loss,"baseline_development":baseline_dev,
                "history":history,"curriculum_source":"mjon-author-written-cc0",
                "teacher_training":"none; curated knowledge provided by MJON developers"}
    if not args.resume:safe_save(snapshot(0),out)
    print(f"MJON v1.1: {len(pools['knowledge'])} ejemplos de conocimiento, "
          f"{len(pools['knowledge_dev'])} paráfrasis de desarrollo; "
          f"base {baseline_dev}, knowledge loss {baseline_loss}.",flush=True)
    for step in range(start+1,args.steps+1):
        student.train();optimizer.zero_grad(set_to_none=True)
        chosen=[]
        for _ in range(args.batch_size):
            roll=float(torch.rand(()))
            kind=("knowledge" if roll<args.knowledge_fraction else
                  "memory" if roll<args.knowledge_fraction+args.memory_fraction else "school")
            subset=pools[kind]
            index=int(torch.randint(len(subset),(1,)))
            chosen.append((kind,subset[index]))
        x,y=collate([record for _,record in chosen])
        logits,_=student(x)
        supervised=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                                   y.reshape(-1),ignore_index=-100)
        replay_indices=[i for i,(key,_) in enumerate(chosen) if key!="knowledge"]
        kd=logits.sum()*0
        if replay_indices and args.kd_weight>0:
            with torch.no_grad():old_logits,_=reference(x[replay_indices])
            kd=frozen_distribution_loss(logits[replay_indices],old_logits,y[replay_indices])
        loss=supervised+args.kd_weight*kd
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            knowledge_loss=evaluate(student,pools["knowledge_dev"],count=len(pools["knowledge_dev"]))
            current_dev=evaluate_development(student,codec,dev)
            eligible=choose_candidate(baseline_dev,current_dev,baseline_loss,knowledge_loss)
            promoted=eligible and knowledge_loss<best_loss-.00001
            if promoted:
                best_loss=knowledge_loss
                best_step=step
            metric={"step":step,"train_loss":round(float(supervised.item()),5),
                    "kd_loss":round(float(kd.item()),5),
                    "knowledge_dev_loss":knowledge_loss,
                    "development":current_dev,"eligible":eligible,"promoted":promoted}
            history.append(metric)
            if promoted:safe_save(snapshot(step),out)
            print("MJON v1.1:",metric,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(snapshot(step),latest)
    best=torch.load(out,map_location="cpu",weights_only=True)
    out.with_suffix(".metrics.json").write_text(json.dumps(
        {"selected_step":best["step"],"base":baseline_dev,
         "knowledge_baseline_loss":baseline_loss,"history":history},
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"MJON v1.1: checkpoint seleccionado {best['step']} / {args.steps}. "
          f"No se promocionan pérdidas de memoria/escuela en calibración.",flush=True)
    return out


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="checkpoints/web/v10.pt")
    p.add_argument("--knowledge",default="data/knowledge_v11/train.json")
    p.add_argument("--paraphrase",default="data/knowledge_v11/paraphrase.json")
    p.add_argument("--unknown",default="data/knowledge_v11/unknown.json")
    p.add_argument("--school",default="data/school_v07/train.json")
    p.add_argument("--memory",default="data/memory_v06/train.json")
    p.add_argument("--school-holdout",default="data/school_v07/holdout.json")
    p.add_argument("--school-transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory-holdout",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="checkpoints/web/v11.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=420)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--knowledge-fraction",type=float,default=.42)
    p.add_argument("--memory-fraction",type=float,default=.23)
    p.add_argument("--kd-weight",type=float,default=.55)
    p.add_argument("--lr",type=float,default=.00008)
    p.add_argument("--eval-every",type=int,default=70)
    p.add_argument("--save-every",type=int,default=70)
    p.add_argument("--seed",type=int,default=1111)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":run(parse_args())
