"""MJON IA v0.8: respuesta-supervisada a partir de profesor abierto.

No copia pesos ni vocabulario de Qwen: entrena parámetros MJON originales
con ejemplos generados por un maestro y repaso de tareas previas.
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
from .teacher_v08 import verify_separation
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save


def prepare(base, teacher_file, replay_file, memory_file):
    checkpoint=torch.load(base,map_location="cpu",weights_only=True)
    if checkpoint.get("version")!=7 or "tokenizer" not in checkpoint:
        raise ValueError("El alumno debe partir de MJON IA v0.7")
    teacher=json.loads(Path(teacher_file).read_text(encoding="utf-8"))
    if teacher.get("schema")!="mjonia.teacher.v1" or teacher.get("teacher_weight_license")!="Apache-2.0":
        raise ValueError("Falta procedencia del profesor y su licencia")
    train,holdout=teacher["training"],teacher["holdout"]
    verify_separation(train,holdout)
    for row in train+holdout:
        if not isinstance(row.get("answer"),str) or not row["answer"].strip():
            raise ValueError("Respuestas de profesor inválidas")
    _,school=read_examples(replay_file)
    _,memory=read_examples(memory_file)
    codec=ByteBPETokenizer.from_dict(checkpoint["tokenizer"])
    cfg=ModelConfig(**checkpoint["model_config"])
    pools={}
    for key, rows in (("teacher",train),("holdout",holdout),("school",school),("memory",memory)):
        records=[]
        for row in rows:
            try:
                records.append(encode_record(codec,row["question"],row["answer"],cfg.context))
            except ValueError:
                # No truncar etiquetas o prompts de forma silenciosa; contabilizar pérdida.
                continue
        if not records:
            raise ValueError(f"No hay ejemplos compatibles con contexto para {key}")
        pools[key]=records
    if len(pools["teacher"])<min(12,len(train)):
        raise ValueError("Demasiados ejemplos perdidos por límite de contexto")
    return checkpoint,teacher,pools


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError("Pasos/lotes inválidos")
    if not 0 < args.teacher_fraction < 1 or not 0 <= args.memory_fraction < 1-args.teacher_fraction:
        raise ValueError("Pesos de entrenamiento inválidos")
    if args.lr<=0:raise ValueError("Tasa de aprendizaje inválida")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    base,manifest,pools=prepare(args.base,args.teacher,args.replay,args.memory)
    cfg=ModelConfig(**base["model_config"])
    model=ByteTransformer(cfg);model.load_state_dict(base["state_dict"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    meta={
        "base_sha256":digest(args.base),"teacher_sha256":digest(args.teacher),
        "replay_sha256":digest(args.replay),"memory_sha256":digest(args.memory),
        "seed":args.seed,"batch_size":args.batch_size,
        "teacher_fraction":args.teacher_fraction,"memory_fraction":args.memory_fraction,
        "lr":args.lr
    }
    first=0;history=[]
    if args.resume:
        previous=torch.load(args.resume,map_location="cpu",weights_only=True)
        if previous.get("version")!=8 or any(previous.get(k)!=v for k,v in meta.items()):
            raise ValueError("Checkpoint de destilación incompatible")
        if previous["model_config"]!=cfg.to_dict() or previous["tokenizer"]!=base["tokenizer"]:
            raise ValueError("Configuración/Tokenizador incompatibles")
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        torch.set_rng_state(previous["torch_rng_state"])
        first=int(previous["step"]);history=list(previous["history"])
        if args.steps<=first:raise ValueError("El paso objetivo debe superar checkpoint")
    print(f"MJON IA v0.8: maestro={manifest['teacher_id']} {manifest['teacher_revision']}; "
          f"datos: {len(pools['teacher'])} docente, {len(pools['holdout'])} reservado, "
          f"{len(pools['school'])} repaso escolar, {len(pools['memory'])} memoria",
          flush=True)
    output=Path(args.out)
    for step in range(first+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        chosen=[]
        for _ in range(args.batch_size):
            random_value=torch.rand(()).item()
            key=("teacher" if random_value<args.teacher_fraction else
                 "memory" if random_value<args.teacher_fraction+args.memory_fraction else "school")
            pool=pools[key]
            ix=int(torch.randint(len(pool),(1,)).item())
            chosen.append(pool[ix])
        x,y=collate(chosen)
        logits,_=model(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            item={"step":step,"train_loss":round(float(loss.item()),5),
                  "teacher_holdout_loss":evaluate(model,pools["holdout"],count=len(pools["holdout"])),
                  "school_replay_loss":evaluate(model,pools["school"],count=20)}
            history.append(item);print("Métricas destilación:",item,flush=True)
        if step%args.save_every==0 or step==args.steps:
            state={"version":8,**meta,"model_config":cfg.to_dict(),
                   "tokenizer":base["tokenizer"],"state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict(),
                   "torch_rng_state":torch.get_rng_state(),"step":step,"history":history,
                   "teacher_id":manifest["teacher_id"],
                   "teacher_revision":manifest["teacher_revision"]}
            safe_save(state,output)
            output.with_suffix(".metrics.json").write_text(
                json.dumps(history,indent=2)+"\n",encoding="utf-8")
    return output


def parse_args():
    p=argparse.ArgumentParser(description="MJON IA v0.8: destilar profesor abierto")
    p.add_argument("--base",default="checkpoints/web/v07.pt")
    p.add_argument("--teacher",default="data/teacher_v08.json")
    p.add_argument("--replay",default="data/school_v07/train.json")
    p.add_argument("--memory",default="data/memory_v06/train.json")
    p.add_argument("--out",default="checkpoints/web/v08.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=480)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--teacher-fraction",type=float,default=.70)
    p.add_argument("--memory-fraction",type=float,default=.12)
    p.add_argument("--lr",type=float,default=.00012)
    p.add_argument("--eval-every",type=int,default=120)
    p.add_argument("--save-every",type=int,default=120)
    p.add_argument("--seed",type=int,default=808)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
