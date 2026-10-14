"""MJON IA: entrenamiento escalable desde pesos aleatorios, sin modelo externo.

El entrenamiento del Neural Core no altera los checkpoints ni el motor web
legacy. Consume corpus JSONL con procedencia, licencia y grupos documentales;
crea BPE SentencePiece con texto SOLO de entrenamiento; separación por
documento y deduplicación; CUDA BF16/FP16 o CPU; guardado atómico reanudable.
"""
import argparse
import contextlib
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from .neural_core import NeuralConfig, MJONNeuralCore
from .train_v03 import safe_save


LICENSES = frozenset({
    "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0",
    "MIT", "Apache-2.0", "public-domain",
})


def split_documents(path, seed=42, fraction=.95):
    if not 0 < fraction < 1:
        raise ValueError("Fracción de split inválida")
    groups={}
    seen=set()
    payload=Path(path).read_bytes()
    for no,line in enumerate(payload.decode("utf-8").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        text=row.get("text")
        owner=row.get("source_id")
        group=row.get("document_id")
        license_name=row.get("license")
        if not all(isinstance(v,str) and v.strip() for v in (text,owner,group,license_name)):
            raise ValueError(f"Faltan metadatos de procedencia en la línea {no}")
        if license_name not in LICENSES:
            raise ValueError(f"Licencia desconocida o no autorizada: {license_name}")
        if not 10 <= len(text) <= 1_000_000:
            raise ValueError(f"Longitud de documento inválida en línea {no}")
        canonical=" ".join(text.casefold().split())
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        # Identificador agrupado por fuente para reducir fuga entre partes de documentos.
        key=owner+"/"+group
        groups.setdefault(key,[]).append(text)
    if len(groups)<8:raise ValueError("Hacen falta al menos ocho documentos diferentes")
    ids=sorted(groups)
    random.Random(seed).shuffle(ids)
    n=max(1,min(len(ids)-1,round(len(ids)*fraction)))
    train=[t for key in ids[:n] for t in groups[key]]
    validation=[t for key in ids[n:] for t in groups[key]]
    if not train or not validation:raise ValueError("Split vacío")
    signature=hashlib.sha256(payload).hexdigest()
    return signature,train,validation,ids[:n],ids[n:]


def create_tokenizer(texts,path,vocab_size,cpu_threads):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Instalá la dependencia opcional: pip install sentencepiece") from exc
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            input_path=Path(folder)/"train_only.txt"
            input_path.write_text("\n".join(t.replace("\n"," ") for t in texts)+"\n",
                                  encoding="utf-8")
            prefix=Path(folder)/"mjon-bpe"
            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(prefix),
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                hard_vocab_limit=False,
                num_threads=cpu_threads,
                unk_id=0,bos_id=1,eos_id=2,pad_id=-1,
                normalization_rule_name="identity",
            )
            path.write_bytes(prefix.with_suffix(".model").read_bytes())
    processor=spm.SentencePieceProcessor(model_file=str(path))
    if processor.vocab_size()<256:
        raise ValueError("Vocabulario real demasiado pequeño")
    return processor, hashlib.sha256(path.read_bytes()).hexdigest()


def encode_documents(processor,documents):
    ids=[]
    for doc in documents:
        ids.extend(processor.encode(doc, out_type=int))
        ids.append(processor.eos_id())
    return torch.tensor(ids,dtype=torch.long)


def batch(tokens,batch_size,context,device):
    max_start=len(tokens)-context-1
    if max_start<1:raise ValueError("Tokens insuficientes para el contexto pedido")
    starts=torch.randint(0,max_start+1,(batch_size,))
    a=torch.stack([tokens[i:i+context] for i in starts.tolist()])
    b=torch.stack([tokens[i+1:i+context+1] for i in starts.tolist()])
    return a.to(device),b.to(device)


@torch.no_grad()
def validation(model,tokens,context,device,precision="fp32",windows=8):
    model.eval()
    n=len(tokens)-context-1
    if n<0:raise ValueError("Datos de validación demasiado pequeños")
    indices=torch.linspace(0,n,steps=min(windows,n+1)).long().tolist()
    total=0.
    for start in indices:
        x=tokens[start:start+context].unsqueeze(0).to(device)
        y=tokens[start+1:start+context+1].unsqueeze(0).to(device)
        with autocast_for(device,precision):
            _,loss,_=model(x,targets=y)
        total+=float(loss)
    return total/len(indices)


def autocast_for(device,precision):
    if device.type!="cuda" or precision=="fp32":
        return contextlib.nullcontext()
    dtype=torch.bfloat16 if precision=="bf16" else torch.float16
    return torch.autocast(device_type="cuda",dtype=dtype)


def learning_rate(step,max_steps,base,warmup,min_factor=.1):
    if warmup<0 or warmup>=max_steps:raise ValueError("Warmup inválido")
    if step<=warmup and warmup>0:
        return base * step/warmup
    progress=min(1.,(step-warmup)/max(1,max_steps-warmup))
    return base*(min_factor+(1-min_factor)*.5*(1+math.cos(math.pi*progress)))


def run(args):
    if min(args.steps,args.context,args.batch_size,args.grad_accum,args.cpu_threads,
           args.eval_every,args.save_every,args.vocab_size)<1:
        raise ValueError("Argumentos del entrenamiento inválidos")
    if args.lr<=0 or args.clip_grad<=0 or not 0<=args.dropout<1:
        raise ValueError("LR, clipping o dropout inválidos")
    if args.precision not in ("fp32","bf16","fp16"):
        raise ValueError("Precisión no admitida")
    device_name=args.device if args.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name=="cuda" and not torch.cuda.is_available():
        raise ValueError("No se detectó GPU CUDA; no se simulará entrenamiento GPU")
    device=torch.device(device_name)
    if device.type=="cpu" and args.precision!="fp32":
        raise ValueError("En CPU solo se admite fp32")
    if device.type=="cuda" and args.precision=="bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("La GPU no admite BF16; probá fp16")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(args.seed)
    signature,docs,val_docs,train_groups,val_groups=split_documents(args.data,args.seed)
    tokenizer,tokenizer_hash=create_tokenizer(docs,args.tokenizer,args.vocab_size,args.cpu_threads)
    train=encode_documents(tokenizer,docs)
    val=encode_documents(tokenizer,val_docs)
    config=NeuralConfig(
        vocab_size=tokenizer.vocab_size(),context=args.context,width=args.width,
        layers=args.layers,heads=args.heads,kv_heads=args.kv_heads,dropout=args.dropout)
    if min(len(train),len(val))<=config.context+1:
        raise ValueError("Corpus de entrenamiento o validación insuficiente")
    model=MJONNeuralCore(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and args.precision=="fp16")
    output=Path(args.out)
    latest=output.with_name(output.stem+".last.pt")
    best_score=float("inf")
    best_step=0
    start=0
    history=[]
    # Se verifica la procedencia y la forma antes de restaurar el optimizador.
    setup={"dataset_sha256":signature,"tokenizer_sha256":tokenizer_hash,
           "seed":args.seed,"config":config.to_dict(),
           "batch_size":args.batch_size,"grad_accum":args.grad_accum,
           "lr":args.lr,"weight_decay":args.weight_decay,"clip_grad":args.clip_grad,
           "precision":args.precision,"warmup":args.warmup,"steps":args.steps}
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=True)
        if ck.get("format")!="mjon-neural-v2" or any(ck.get(k)!=v for k,v in setup.items()):
            raise ValueError("El checkpoint no coincide con los datos o hiperparámetros")
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng_states"])
        start=int(ck["step"])
        best_score=float(ck["best_val_loss"])
        best_step=int(ck["best_step"])
        history=list(ck["history"])
        if start>=args.steps:raise ValueError("El checkpoint ya llegó a la meta")
    def checkpoint(step):
        data={
            "format":"mjon-neural-v2",**setup,
            "state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scaler_state_dict":scaler.state_dict(),
            "torch_rng_state":torch.get_rng_state(),
            "step":step,"best_step":best_step,"best_val_loss":best_score,
            "history":history,"total_train_tokens":len(train),"total_valid_tokens":len(val),
            "train_groups_sha256":hashlib.sha256(json.dumps(train_groups).encode()).hexdigest(),
            "validation_groups_sha256":hashlib.sha256(json.dumps(val_groups).encode()).hexdigest()
        }
        if device.type=="cuda":
            data["cuda_rng_states"]=torch.cuda.get_rng_state_all()
        return data
    print(f"MJON Neural Core: {model.parameter_count():,} parámetros, "
          f"{len(train):,} tokens train, {len(val):,} val, device={device}, "
          f"precision={args.precision}",flush=True)
    for step in range(start+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(step,args.steps,args.lr,args.warmup)
        for group in optimizer.param_groups:group["lr"]=lr
        running=0.0
        for _ in range(args.grad_accum):
            x,y=batch(train,args.batch_size,config.context,device)
            with autocast_for(device,args.precision):
                _,loss,_=model(x,targets=y)
                part=loss/args.grad_accum
            scaler.scale(part).backward()
            running+=float(loss.detach())/args.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            val_loss=validation(model,val,config.context,device,args.precision)
            record={"step":step,"loss":round(running,5),"val_loss":round(val_loss,5),
                    "val_ppl":round(math.exp(min(val_loss,20.)),2),"lr":lr,
                    "tokens_processed":step*args.batch_size*args.context*args.grad_accum}
            history.append(record)
            if val_loss<best_score-1e-5:
                best_score=val_loss
                best_step=step
                safe_save(checkpoint(step),output)
            print("MJON Neural métricas:",record,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(checkpoint(step),latest)
    output.with_suffix(".metrics.json").write_text(
        json.dumps({"best_step":best_step,"best_val_loss":best_score,
                    "history":history,"model_config":config.to_dict()},
                   ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return output


def parse_args():
    p=argparse.ArgumentParser(description="Entrenamiento escalable de MJON Original desde cero")
    p.add_argument("--data",required=True,help="JSONL con text, source_id, document_id, license")
    p.add_argument("--tokenizer",default="checkpoints/neural/mjon-bpe.model")
    p.add_argument("--out",default="checkpoints/neural/mjon-neural.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--vocab-size",type=int,default=4096)
    p.add_argument("--context",type=int,default=1024)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--layers",type=int,default=8)
    p.add_argument("--heads",type=int,default=6)
    p.add_argument("--kv-heads",type=int,default=2)
    p.add_argument("--dropout",type=float,default=0.)
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--grad-accum",type=int,default=8)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--weight-decay",type=float,default=.1)
    p.add_argument("--clip-grad",type=float,default=1.)
    p.add_argument("--warmup",type=int,default=100)
    p.add_argument("--eval-every",type=int,default=200)
    p.add_argument("--save-every",type=int,default=200)
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--precision",choices=["fp32","bf16","fp16"],default="fp32")
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
