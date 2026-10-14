"""MJON Original v0.10: aprendizaje continuo contra el olvido catastrófico.

Parte de v0.8 (antes de la regresión v0.9), enseña respuestas Qwen y
protege la distribución de salida original en ejemplos escolares/memoria.
Se publica el checkpoint que supera criterios sin perder exactitud base.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .finetune_v05 import collate, encode_record, evaluate
from .finetune_v06 import read_examples
from .model import ByteTransformer, ModelConfig
from .teacher_v08 import verify_separation
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save


def load_data(args):
    baseline = torch.load(args.base, map_location="cpu", weights_only=True)
    if baseline.get("version") != 8:
        raise ValueError("La base de v0.10 es el Transformer MJON v0.8")
    teaching = json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    if teaching.get("generation_version") != 9 or teaching.get("teacher_weight_license") != "Apache-2.0":
        raise ValueError("Faltan metadatos del maestro")
    verify_separation(teaching["training"], teaching["holdout"])
    codec = ByteBPETokenizer.from_dict(baseline["tokenizer"])
    context = baseline["model_config"]["context"]
    _, school = read_examples(args.replay)
    _, memory = read_examples(args.memory)
    _, school_reserved = read_examples(args.school_holdout)
    _, school_transfer = read_examples(args.school_transfer)
    _, memory_reserved = read_examples(args.memory_holdout)
    if len(school_reserved) < 48 or len(school_transfer) < 48:
        raise ValueError("Se requieren preguntas de calibración distintas del examen final")
    pools = {}
    for name, rows in (("teacher", teaching["training"]),
                       ("teacher_validation", teaching["holdout"][:max(1,len(teaching["holdout"])//2)]),
                       ("school", school), ("memory", memory)):
        encoded=[]
        for row in rows:
            try:
                encoded.append(encode_record(codec,row["question"],row["answer"],context))
            except ValueError:
                if name.startswith("teacher"):
                    raise
        if not encoded:raise ValueError(f"Corpus vacío: {name}")
        pools[name]=encoded
    # Las pruebas oficiales anteriores toman los primeros 10 de cada materia.
    # La calibración toma 10:14. Nunca usar este conjunto para gradientes.
    school_development=[]
    transfer_development=[]
    for area in ("matematica","lengua","ciencias","logica"):
        school_development.extend([x for x in school_reserved if x["subject"]==area][10:14])
        transfer_development.extend([x for x in school_transfer if x["subject"]==area][10:14])
    if len(school_development)!=16 or len(transfer_development)!=16:
        raise ValueError("Datos de calibración incompletos")
    memory_development=[r for r in memory_reserved if r.get("case") in ("recall","correction")][:8]
    if len(memory_development)!=8:raise ValueError("Calibración de memoria incompleta")
    development=school_development+transfer_development+memory_development
    # Validar exclusión de desarrollo del entrenamiento: igualdad exacta.
    train_questions={r["question"] for r in teaching["training"]+school+memory}
    if any(x["question"] in train_questions for x in development):
        raise ValueError("Fuga de preguntas de calibración al entrenamiento")
    return baseline,teaching,codec,pools,development


@torch.no_grad()
def answer(model,codec,question,limit=16):
    model.eval()
    ids=codec.encode(f"Usuario: {question}\nAsistente:")
    output=[]
    for _ in range(limit):
        inp=torch.tensor([ids[-model.config.context:]],dtype=torch.long)
        logits,_=model(inp)
        next_token=int(logits[0,-1].argmax())
        ids.append(next_token)
        output.append(next_token)
        decoded=codec.decode(output)
        if "\n" in decoded:return decoded.split("\n",1)[0]
    return codec.decode(output)


def normalize(value):
    return " ".join(value.strip().lower().rstrip(".!?¿¡ ").split())


@torch.no_grad()
def evaluate_development(model,codec,rows):
    # Comparación con salidas generadas, no con elección forzada de respuestas.
    successes=[]
    for row in rows:
        pred=answer(model,codec,row["question"])
        successes.append(normalize(pred)==normalize(row["answer"]))
    return {
        "school":sum(successes[:16]),"transfer":sum(successes[16:32]),
        "memory":sum(successes[32:]),"total":sum(successes),
    }


def choose_candidate(baseline_metrics, candidate_metrics, baseline_loss, candidate_loss,
                     tolerance=0):
    """Gate duro de retención y mejora del profesor; no hay promoción automática."""
    if candidate_metrics["school"] < baseline_metrics["school"]-tolerance:return False
    if candidate_metrics["memory"] < baseline_metrics["memory"]-tolerance:return False
    if candidate_metrics["transfer"] < baseline_metrics["transfer"]-tolerance:return False
    if candidate_loss >= baseline_loss - .01:return False
    return True


def frozen_distribution_loss(student_logits, reference_logits, labels, temperature=2.):
    """KL sobre los tokens de respuesta en datos antiguos; teacher = MJON v0.8."""
    mask=(labels!=-100)
    if not mask.any():return student_logits.sum()*0
    raw=student_logits[mask]/temperature
    ref=reference_logits[mask]/temperature
    return F.kl_div(F.log_softmax(raw,dim=-1),
                    F.softmax(ref,dim=-1),reduction="batchmean")*(temperature**2)


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError("Pasos y lotes deben ser positivos")
    if not 0<args.teacher_fraction<1 or not 0<args.memory_fraction<1-args.teacher_fraction:
        raise ValueError("Mezcla docente/repaso inválida")
    if not 0<=args.kd_weight<=10 or args.lr<=0:
        raise ValueError("Pérdida KD o lr inválidos")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    base,teacher,codec,pools,development=load_data(args)
    config=ModelConfig(**base["model_config"])
    model=ByteTransformer(config)
    model.load_state_dict(base["state_dict"])
    reference=ByteTransformer(config).eval()
    reference.load_state_dict(base["state_dict"])
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    signature={
        "base_sha256":hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "teacher_sha256":hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest(),
        "school_sha256":hashlib.sha256(Path(args.replay).read_bytes()).hexdigest(),
        "memory_sha256":hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
        "school_holdout_sha256":hashlib.sha256(Path(args.school_holdout).read_bytes()).hexdigest(),
        "transfer_sha256":hashlib.sha256(Path(args.school_transfer).read_bytes()).hexdigest(),
        "memory_holdout_sha256":hashlib.sha256(Path(args.memory_holdout).read_bytes()).hexdigest(),
        "seed":args.seed,"lr":args.lr,"batch_size":args.batch_size,
        "teacher_fraction":args.teacher_fraction,"memory_fraction":args.memory_fraction,
        "kd_weight":args.kd_weight,"eval_every":args.eval_every,
    }
    best_path=Path(args.out)
    last_path=best_path.with_name(best_path.stem+".last.pt")
    baseline_dev=evaluate_development(reference,codec,development)
    baseline_loss=evaluate(reference,pools["teacher_validation"],
                           count=len(pools["teacher_validation"]))
    current_best_loss=baseline_loss
    current_best_step=0
    history=[]
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=True)
        if state.get("version")!=10 or any(state.get(k)!=v for k,v in signature.items()):
            raise ValueError("Checkpoint incompatible con los datos o entrenamiento")
        if state["model_config"]!=base["model_config"] or state["tokenizer"]!=base["tokenizer"]:
            raise ValueError("Modelo/tokenizador incompatible")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        history=list(state["history"])
        current_best_loss=float(state["best_teacher_loss"])
        current_best_step=int(state["best_step"])
        start=int(state["step"])
        if args.steps<=start:raise ValueError("steps debe superar último checkpoint")
    else:
        history=[{"step":0,"teacher_validation_loss":baseline_loss,
                  "development":baseline_dev,"promoted":True}]
        print("MJON Original base:",history[-1],flush=True)
    def snapshot(step):
        return {"version":10,**signature,"model_config":base["model_config"],
                "tokenizer":base["tokenizer"],"state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"step":step,
                "history":history,"baseline_development":baseline_dev,
                "baseline_teacher_loss":baseline_loss,
                "best_teacher_loss":current_best_loss,
                "best_step":current_best_step,
                "teacher_id":teacher["teacher_id"],
                "teacher_revision":teacher["teacher_revision"]}
    if not args.resume:safe_save(snapshot(0),best_path)
    for step in range(start+1,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_group=[]
        # Clasificación controlada: KD únicamente sobre tokens antiguos.
        for _ in range(args.batch_size):
            r=float(torch.rand(()))
            group=("teacher" if r<args.teacher_fraction else
                   "memory" if r<args.teacher_fraction+args.memory_fraction else "school")
            candidates=pools[group]
            idx=int(torch.randint(len(candidates),(1,)))
            data_group.append((group,candidates[idx]))
        tokens,labels=collate([row for _,row in data_group])
        logits,_=model(tokens)
        supervised=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                                   labels.reshape(-1),ignore_index=-100)
        indices=[i for i,(group,_) in enumerate(data_group) if group!="teacher"]
        kd=logits.sum()*0
        if indices and args.kd_weight>0:
            with torch.no_grad():
                old_logits,_=reference(tokens[indices])
            kd=frozen_distribution_loss(logits[indices],old_logits,labels[indices])
        loss=supervised + args.kd_weight*kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            val_loss=evaluate(model,pools["teacher_validation"],
                              count=len(pools["teacher_validation"]))
            dev=evaluate_development(model,codec,development)
            allowed=choose_candidate(baseline_dev,dev,baseline_loss,val_loss)
            promoted=allowed and val_loss<current_best_loss-.00001
            if promoted:
                current_best_loss=val_loss
                current_best_step=step
            item={"step":step,"train_loss":round(float(supervised.item()),5),
                  "kd_loss":round(float(kd.item()),5),
                  "teacher_validation_loss":val_loss,
                  "development":dev,"eligible":allowed,"promoted":promoted}
            history.append(item)
            if promoted:safe_save(snapshot(step),best_path)
            print("MJON v0.10:",item,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(snapshot(step),last_path)
    best=torch.load(best_path,map_location="cpu",weights_only=True)
    best_path.with_suffix(".metrics.json").write_text(
        json.dumps({"baseline":baseline_dev,"baseline_teacher_loss":baseline_loss,
                    "selected_step":best["step"],"history":history},
                   indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"MJON v0.10: publicado paso {best['step']} de {args.steps}; "
          f"profesor loss {best['best_teacher_loss']} vs base {baseline_loss}, "
          f"no se acepta regresión de calibración.",flush=True)
    return best_path


def parse_args():
    p=argparse.ArgumentParser(description="MJON original: actualización con distilación de conocimiento y protección antiolvido")
    p.add_argument("--base",default="checkpoints/web/v08.pt")
    p.add_argument("--teacher",default="data/teacher_v09.json")
    p.add_argument("--replay",default="data/school_v07/train.json")
    p.add_argument("--memory",default="data/memory_v06/train.json")
    p.add_argument("--school-holdout",default="data/school_v07/holdout.json")
    p.add_argument("--school-transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory-holdout",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="checkpoints/web/v10.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=350)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--teacher-fraction",type=float,default=.30)
    p.add_argument("--memory-fraction",type=float,default=.25)
    p.add_argument("--kd-weight",type=float,default=.6)
    p.add_argument("--lr",type=float,default=.00005)
    p.add_argument("--eval-every",type=int,default=70)
    p.add_argument("--save-every",type=int,default=70)
    p.add_argument("--seed",type=int,default=1010)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":run(parse_args())
