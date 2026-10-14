"""MJON Escuela v0.7: ajuste multiasignatura desde el Transformer v0.6.

Conserva vocabulario y pesos neuronales; duplica contexto 128 -> 256
ampliando embeddings posicionales, y mezcla tareas anteriores para retención.
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
from .train_v03 import safe_save


def upgrade_model(base, context):
    old_cfg = ModelConfig(**base["model_config"])
    if context < old_cfg.context:
        raise ValueError("No se admite reducir el contexto")
    cfg = ModelConfig(**{**old_cfg.to_dict(), "context": context})
    model = ByteTransformer(cfg)
    old_state = base["state_dict"]
    state = model.state_dict()
    for key,value in state.items():
        previous = old_state[key]
        if key == "position.weight" and value.shape != previous.shape:
            old_positions = previous.shape[0]
            for index in range(context):
                value[index].copy_(previous[index % old_positions])
        else:
            if value.shape != previous.shape:
                raise ValueError(f"Tensor incompatible: {key}")
            value.copy_(previous)
    model.load_state_dict(state)
    return model


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads,args.context)<1:
        raise ValueError("Parámetros de entrenamiento inválidos")
    if args.lr<=0 or args.replay_v05<0 or args.replay_v06<0 or args.replay_v05+args.replay_v06>=1:
        raise ValueError("Proporciones de repaso o learning rate inválidas")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    base_file=Path(args.base)
    base_bytes=base_file.read_bytes()
    base_hash=hashlib.sha256(base_bytes).hexdigest()
    base=torch.load(base_file,map_location="cpu",weights_only=True)
    if base.get("version")!=6 or not base.get("tokenizer"):
        raise ValueError("El modelo base debe ser v0.6 propio con BPE")
    codec=ByteBPETokenizer.from_dict(base["tokenizer"])
    model=upgrade_model(base,args.context)
    cfg=model.config
    specs={}
    pools={}
    for name,path in (("train",args.train),("holdout",args.holdout),("transfer",args.transfer),
                      ("replay05",args.replay05),("replay06",args.replay06)):
        digest,records=read_examples(path)
        specs[name]=digest
        pools[name]=[encode_record(codec,r["question"],r["answer"],cfg.context) for r in records]
    if len({r["question"] for r in json.loads(Path(args.train).read_text(encoding="utf-8"))} &
           {r["question"] for r in json.loads(Path(args.holdout).read_text(encoding="utf-8"))}):
        raise ValueError("Entrenamiento contaminado con holdout")
    if not pools["transfer"] or not pools["holdout"]:
        raise ValueError("Evaluaciones vacías")
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    history=[]
    first=0
    signature={"base_sha256":base_hash,"data_sha256":specs,"seed":args.seed,
               "learning_rate":args.lr,"batch_size":args.batch_size,
               "replay_v05":args.replay_v05,"replay_v06":args.replay_v06}
    if args.resume:
        previous=torch.load(args.resume,map_location="cpu",weights_only=True)
        if previous.get("version")!=7 or any(previous.get(k)!=v for k,v in signature.items()):
            raise ValueError("Checkpoint v0.7 incompatible con base, datos o hiperparámetros")
        if previous["model_config"]!=cfg.to_dict() or previous["tokenizer"]!=codec.to_dict():
            raise ValueError("Arquitectura incompatible")
        model.load_state_dict(previous["state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        torch.set_rng_state(previous["torch_rng_state"])
        first=int(previous["step"])
        history=list(previous["history"])
        if args.steps<=first:raise ValueError("--steps debe superar el paso guardado")
    print(f"MJON Escuela v0.7: {sum(p.numel() for p in model.parameters()):,} parámetros; "
          f"contexto {cfg.context}; corpus {len(pools['train'])} y "
          f"holdout {len(pools['holdout'])} / transfer {len(pools['transfer'])}",flush=True)
    output=Path(args.out)
    for step in range(first+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=[]
        for _ in range(args.batch_size):
            choice=torch.rand(()).item()
            pool=("replay05" if choice<args.replay_v05 else
                  "replay06" if choice<args.replay_v05+args.replay_v06 else "train")
            pool_records=pools[pool]
            index=int(torch.randint(len(pool_records),(1,)).item())
            batch.append(pool_records[index])
        x,y=collate(batch)
        logits,_=model(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                             y.reshape(-1),ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            metric={"step":step,"train_loss":round(loss.item(),5),
                    "holdout_loss":evaluate(model,pools["holdout"],count=24),
                    "transfer_loss":evaluate(model,pools["transfer"],count=24)}
            history.append(metric);print("Métricas:",metric,flush=True)
        if step%args.save_every==0 or step==args.steps:
            checkpoint={"version":7,**signature,"model_config":cfg.to_dict(),
                        "tokenizer":codec.to_dict(),"state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "torch_rng_state":torch.get_rng_state(),
                        "step":step,"history":history}
            safe_save(checkpoint,output)
            output.with_suffix(".metrics.json").write_text(
                json.dumps(history,indent=2)+"\n",encoding="utf-8")
    return output


def parse_args():
    p=argparse.ArgumentParser(description="MJON Escuela v0.7")
    p.add_argument("--base",default="checkpoints/web/v06.pt")
    p.add_argument("--train",default="data/school_v07/train.json")
    p.add_argument("--holdout",default="data/school_v07/holdout.json")
    p.add_argument("--transfer",default="data/school_v07/transfer.json")
    p.add_argument("--replay05",default="data/instruction_v05/train.json")
    p.add_argument("--replay06",default="data/memory_v06/train.json")
    p.add_argument("--out",default="checkpoints/web/v07.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--context",type=int,default=256)
    p.add_argument("--steps",type=int,default=1100)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--lr",type=float,default=.0002)
    p.add_argument("--replay-v05",type=float,default=.12)
    p.add_argument("--replay-v06",type=float,default=.12)
    p.add_argument("--eval-every",type=int,default=220)
    p.add_argument("--save-every",type=int,default=220)
    p.add_argument("--seed",type=int,default=707)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":run(parse_args())
