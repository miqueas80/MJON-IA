"""Ajuste supervisado real: calcula pérdida solo en la respuesta, no en el prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from .model import ByteTransformer,ModelConfig
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save

def encode_record(codec,question,answer,context):
    prefix=codec.encode(f"Usuario: {question}\nAsistente:")
    suffix=codec.encode(f" {answer}\n")
    full=prefix+suffix
    if len(full)>context:raise ValueError(f"Ejemplo supera contexto: {len(full)} > {context}")
    x=full[:-1]
    target=[-100]*(len(prefix)-1)+full[len(prefix):]
    assert len(target)==len(x)
    return x,target

def collate(records):
    length=max(len(x) for x,_ in records)
    x=torch.zeros((len(records),length),dtype=torch.long)
    labels=torch.full((len(records),length),-100,dtype=torch.long)
    for n,(tokens,target) in enumerate(records):
        x[n,:len(tokens)]=torch.tensor(tokens)
        labels[n,:len(target)]=torch.tensor(target)
    return x,labels

@torch.no_grad()
def evaluate(model,records,count=24):
    model.eval();losses=[]
    for item in records[:count]:
        x,y=collate([item])
        logits,_=model(x)
        losses.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100).item())
    return round(sum(losses)/len(losses),5)

def run(args):
    if min(args.steps,args.batch_size,args.grad_accum,args.save_every,args.eval_every,args.cpu_threads)<1:raise ValueError("Argumento no positivo")
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    base=torch.load(args.base,map_location="cpu",weights_only=True)
    if base.get("version")!=3 or not base.get("tokenizer"):raise ValueError("La base debe contener el checkpoint BPE propio v0.4")
    cfg=ModelConfig(**base["model_config"])
    codec=ByteBPETokenizer.from_dict(base["tokenizer"])
    model=ByteTransformer(cfg);model.load_state_dict(base["state_dict"])
    train_json=Path(args.train).read_bytes();val_json=Path(args.holdout).read_bytes()
    train_hash=hashlib.sha256(train_json).hexdigest();val_hash=hashlib.sha256(val_json).hexdigest()
    train_records=json.loads(train_json);validation_records=json.loads(val_json)
    if not set(r["question"] for r in train_records).isdisjoint(r["question"] for r in validation_records):
        raise ValueError("Fuga de ejemplos: preguntas repetidas entre entrenamiento y validación")
    train=[encode_record(codec,r["question"],r["answer"],cfg.context) for r in train_records]
    validation=[encode_record(codec,r["question"],r["answer"],cfg.context) for r in validation_records]
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    first=0;history=[];base_hash=hashlib.sha256(Path(args.base).read_bytes()).hexdigest()
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=True)
        if ck.get("version")!=5 or (ck["base_sha256"],ck["train_sha256"],ck["holdout_sha256"])!=(base_hash,train_hash,val_hash):
            raise ValueError("Checkpoint de ajuste o datasets incompatibles")
        if ck["model_config"]!=cfg.to_dict() or ck["learning_rate"]!=args.lr:raise ValueError("Configuración incompatible")
        model.load_state_dict(ck["state_dict"]);optimizer.load_state_dict(ck["optimizer_state_dict"])
        torch.set_rng_state(ck["torch_rng_state"])
        first=int(ck["step"]);history=list(ck["history"])
        if first>=args.steps:raise ValueError("steps debe superar el paso reanudado")
    print(f"MJON v0.5 | {sum(p.numel() for p in model.parameters())} parámetros | {len(train)} entrenamiento | {len(validation)} reservados",flush=True)
    out=Path(args.out)
    for step in range(first+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            ids=torch.randint(len(train),(args.batch_size,)).tolist()
            x,y=collate([train[i] for i in ids])
            logits,_=model(x)
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
            (loss/args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            point={"step":step,"train_loss":round(loss.item(),5),"holdout_loss":evaluate(model,validation)}
            history.append(point);print("Métricas:",point,flush=True)
        if step%args.save_every==0 or step==args.steps:
            ck={"version":5,"model_config":cfg.to_dict(),"tokenizer":codec.to_dict(),
                "state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"step":step,"history":history,
                "base_sha256":base_hash,"train_sha256":train_hash,"holdout_sha256":val_hash,
                "learning_rate":args.lr}
            safe_save(ck,out)
            out.with_suffix(".metrics.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    return out

def parse_args():
    p=argparse.ArgumentParser(description="MJON IA v0.5: aprendizaje supervisado de respuestas breves")
    p.add_argument("--base",default="checkpoints/web/v04.pt")
    p.add_argument("--train",default="data/instruction_v05/train.json")
    p.add_argument("--holdout",default="data/instruction_v05/holdout.json")
    p.add_argument("--out",default="checkpoints/web/v05.pt")
    p.add_argument("--steps",type=int,default=900)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--eval-every",type=int,default=150)
    p.add_argument("--save-every",type=int,default=150)
    p.add_argument("--lr",type=float,default=.0003)
    p.add_argument("--seed",type=int,default=505)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--resume",default=None)
    return p.parse_args()

if __name__=="__main__":run(parse_args())
