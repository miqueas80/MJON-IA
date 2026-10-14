"""MJON Neural SFT: aprendizaje de instrucciones sobre el Transformer PROPIO.

Carga un checkpoint preentrenado MJON Neural Core, no importa ninguna red
de terceros ni consulta APIs. SFT solo sobre tokens de la respuesta, split
por concepto/documento, guardado reversible y evaluación independiente.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .neural_core import MJONNeuralCore, NeuralConfig
from .scalable_train import LICENSES, autocast_for, learning_rate
from .train_v03 import safe_save


def read_instructions(path, seed=2026, validation_fraction=.15):
    if not 0 < validation_fraction < .5:
        raise ValueError("La fracción de validación debe estar entre 0 y 0.5")
    original=Path(path).read_bytes()
    grouped=defaultdict(list)
    seen_by_question={}
    seen_exact=set()
    for i,raw in enumerate(original.decode("utf-8").splitlines(),1):
        if not raw.strip():continue
        row=json.loads(raw)
        fields=("instruction","response","source_id","group_id","license")
        if any(not isinstance(row.get(f),str) or not row[f].strip() for f in fields):
            raise ValueError(f"Línea {i}: faltan campos de instrucción o procedencia")
        if row["license"] not in LICENSES:
            raise ValueError(f"Línea {i}: licencia no autorizada {row['license']}")
        if not 3<=len(row["instruction"])<=2500 or not 1<=len(row["response"])<=3000:
            raise ValueError(f"Línea {i}: instrucción o respuesta fuera de rango")
        group=row["source_id"].strip()+"/"+row["group_id"].strip()
        normalized=" ".join(row["instruction"].casefold().split())
        owner=seen_by_question.setdefault(normalized,group)
        if owner!=group:
            raise ValueError("Una instrucción aparece en dos grupos: fuga entre entrenamiento y examen")
        key=(group,normalized,row["response"])
        if key in seen_exact:continue
        seen_exact.add(key)
        grouped[group].append(row)
    if len(grouped)<8:
        raise ValueError("Al menos 8 grupos distintos para separar entrenamiento y examen")
    keys=sorted(grouped)
    random.Random(seed).shuffle(keys)
    nval=max(2,min(len(keys)-2,round(len(keys)*validation_fraction)))
    training_keys=set(keys[:-nval])
    validation_keys=set(keys[-nval:])
    assert not(training_keys&validation_keys)
    train=[row for key in keys[:-nval] for row in grouped[key]]
    valid=[row for key in keys[-nval:] for row in grouped[key]]
    return train,valid,{
        "data_sha256":hashlib.sha256(original).hexdigest(),
        "training_group_count":len(training_keys),
        "validation_group_count":len(validation_keys),
        "train_groups_sha256":hashlib.sha256(json.dumps(sorted(training_keys)).encode()).hexdigest(),
        "validation_groups_sha256":hashlib.sha256(json.dumps(sorted(validation_keys)).encode()).hexdigest(),
        "validation_fraction":validation_fraction,
        "seed":seed,
    }


def encode_supervised(tokenizer,row,context):
    prefix=tokenizer.encode("Usuario: "+row["instruction"]+"\nAsistente:",
                            out_type=int)
    answer=tokenizer.encode(" "+row["response"],out_type=int)
    bos=tokenizer.bos_id()
    eos=tokenizer.eos_id()
    full=[bos]+prefix+answer+[eos]
    if len(full)<4 or len(full)>context+1:
        return None
    x=full[:-1]
    y=full[1:]
    # Tokens of prompt ignored, EOS + all answer tokens supervised.
    masked=len(prefix)
    y[:masked]=[-100]*masked
    if all(v==-100 for v in y):
        raise AssertionError("No quedaron tokens de respuesta")
    return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)


def pad_batch(records,device):
    length=max(len(x) for x,_ in records)
    x=torch.zeros((len(records),length),dtype=torch.long)
    labels=torch.full((len(records),length),-100,dtype=torch.long)
    for index,(tokens,targets) in enumerate(records):
        n=len(tokens)
        x[index,:n]=tokens
        labels[index,:n]=targets
    return x.to(device),labels.to(device)


@torch.no_grad()
def validation_loss(model,rows,batch_size,device,precision):
    model.eval()
    weight_sum=0
    loss_sum=0.0
    for offset in range(0,len(rows),batch_size):
        tokens,labels=pad_batch(rows[offset:offset+batch_size],device)
        with autocast_for(device,precision):
            logits,_,_=model(tokens)
            flattened=F.cross_entropy(
                logits.float().reshape(-1,logits.shape[-1]),
                labels.reshape(-1),ignore_index=-100,reduction="sum")
        n=int((labels!=-100).sum().item())
        weight_sum+=n
        loss_sum+=float(flattened)
    if not weight_sum:raise ValueError("Evaluación sin tokens de respuesta")
    return loss_sum/weight_sum


def run(args):
    if min(args.steps,args.batch_size,args.grad_accum,args.eval_every,args.save_every,
           args.cpu_threads)<1:
        raise ValueError("Todos los contadores de entrenamiento deben ser positivos")
    if args.lr<=0 or args.clip_grad<=0:raise ValueError("LR y clipping positivos")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device_name=args.device if args.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible")
    device=torch.device(device_name)
    if device.type=="cpu" and args.precision!="fp32":
        raise ValueError("En CPU se requiere FP32")
    if device.type=="cuda" and args.precision=="bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 no soportado")
    import sentencepiece as spm
    tokenizer_file=Path(args.tokenizer)
    tokenizer_hash=hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    base_path=Path(args.base)
    pretrained=torch.load(base_path,map_location="cpu",weights_only=True)
    if pretrained.get("format")!="mjon-neural-v2":
        raise ValueError("No es un modelo Neural Core propio preentrenado")
    if pretrained.get("tokenizer_sha256")!=tokenizer_hash:
        raise ValueError("Tokenizer diferente al del preentrenamiento")
    config=NeuralConfig(**pretrained["config"])
    if config.vocab_size!=tokenizer.vocab_size():
        raise ValueError("Modelo y tokenizador con vocabulario diferente")
    train_rows,heldout,provenance=read_instructions(args.data,args.seed)
    training=[encode_supervised(tokenizer,row,config.context) for row in train_rows]
    validation=[encode_supervised(tokenizer,row,config.context) for row in heldout]
    train=[row for row in training if row is not None]
    valid=[row for row in validation if row is not None]
    if not train or not valid:
        raise ValueError("Sin suficientes ejemplos que entren en el contexto del modelo")
    if len(train)<max(4,int(len(train_rows)*.75)) or len(valid)<max(2,int(len(heldout)*.75)):
        raise ValueError("Demasiadas respuestas quedaron fuera del contexto; ampliar ventana o mejorar corpus")
    student=MJONNeuralCore(config).to(device)
    student.load_state_dict(pretrained["state_dict"])
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and args.precision=="fp16")
    start=0
    best_step=0
    best_loss=validation_loss(student,valid,args.batch_size,device,args.precision)
    history=[{"step":0,"validation_loss":round(best_loss,6),"promoted":True}]
    checkpoint_path=Path(args.out)
    latest_path=checkpoint_path.with_name(checkpoint_path.stem+".last.pt")
    setup={
        "base_model_sha256":hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "tokenizer_sha256":tokenizer_hash,
        "seed":args.seed,"config":config.to_dict(),**provenance,
        "batch_size":args.batch_size,"grad_accum":args.grad_accum,
        "lr":args.lr,"weight_decay":args.weight_decay,
        "clip_grad":args.clip_grad,"precision":args.precision,
        "steps":args.steps,"warmup":args.warmup,
    }
    if args.resume:
        resume=torch.load(args.resume,map_location="cpu",weights_only=True)
        if resume.get("format")!="mjon-neural-sft-v1" or any(resume.get(k)!=v for k,v in setup.items()):
            raise ValueError("SFT no coincide con corpus, tokenizer o configuración")
        student.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type=="cuda":torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        start=int(resume["step"])
        history=list(resume["history"])
        best_step=int(resume["best_step"])
        best_loss=float(resume["best_validation_loss"])
        if start>=args.steps:raise ValueError("El entrenamiento ya llegó al número de pasos")
    def snapshot(step):
        data={
            "format":"mjon-neural-sft-v1",**setup,
            "state_dict":student.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scaler_state_dict":scaler.state_dict(),
            "torch_rng_state":torch.get_rng_state(),
            "step":step,"best_step":best_step,
            "best_validation_loss":best_loss,"history":history,
            "training_examples":len(train),"validation_examples":len(valid),
            "teacher_weights_used":False,
        }
        if device.type=="cuda":data["cuda_rng_states"]=torch.cuda.get_rng_state_all()
        return data
    if not args.resume:safe_save(snapshot(0),checkpoint_path)
    print(f"MJON Neural SFT: {len(train)}/{len(train_rows)} ejemplos entrenables; "
          f"{len(valid)}/{len(heldout)} reservados; "
          f"pérdida inicial {best_loss:.5f}; parámetros {student.parameter_count():,}",
          flush=True)
    for step in range(start+1,args.steps+1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        rate=learning_rate(step,args.steps,args.lr,args.warmup)
        for group in optimizer.param_groups:group["lr"]=rate
        running=0.
        for _ in range(args.grad_accum):
            idx=torch.randint(len(train),(args.batch_size,)).tolist()
            tokens,labels=pad_batch([train[i] for i in idx],device)
            with autocast_for(device,args.precision):
                logits,_,_=student(tokens)
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                     labels.reshape(-1),ignore_index=-100)
            scaler.scale(loss/args.grad_accum).backward()
            running+=float(loss.detach())/args.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(),args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            measured=validation_loss(student,valid,args.batch_size,device,args.precision)
            promoted=measured<best_loss-1e-5
            if promoted:best_step=step;best_loss=measured
            row={"step":step,"train_loss":round(running,6),
                 "validation_loss":round(measured,6),"promoted":promoted}
            history.append(row)
            if promoted:safe_save(snapshot(step),checkpoint_path)
            print("MJON Neural SFT:",row,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(snapshot(step),latest_path)
    checkpoint_path.with_suffix(".metrics.json").write_text(
        json.dumps({"best_step":best_step,"best_validation_loss":best_loss,
                    "history":history,"provenance":provenance},ensure_ascii=False,
                   indent=2)+"\n",encoding="utf-8")
    return checkpoint_path


def parse_args():
    p=argparse.ArgumentParser(description="MJON Neural Core · ajuste supervisado original")
    p.add_argument("--base",required=True,help="Pesos propios preentrenados")
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--data",required=True,help="JSONL con instruction,response,source_id,group_id,license")
    p.add_argument("--out",default="checkpoints/neural/mjon-instruct.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--grad-accum",type=int,default=4)
    p.add_argument("--lr",type=float,default=0.00008)
    p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--clip-grad",type=float,default=1.)
    p.add_argument("--warmup",type=int,default=100)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--precision",choices=("fp32","bf16","fp16"),default="fp32")
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
