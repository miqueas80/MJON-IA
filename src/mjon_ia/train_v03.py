"""MJON IA v0.3: BPE propio, gradientes acumulados, checkpoints recuperables."""
import argparse
import hashlib
import json
import random
import os
from pathlib import Path
import torch
from .data import random_batch,split_dialogues
from .model import ByteTransformer,ModelConfig
from .tokenizer import ByteBPETokenizer


def prepare_data(path,seed,vocab_size):
    content=Path(path).read_bytes()
    digest=hashlib.sha256(content).hexdigest()
    train_raw,val_raw=split_dialogues(content.decode('utf-8'),seed=seed)
    training=bytes(train_raw.tolist()).decode('utf-8')
    validation=bytes(val_raw.tolist()).decode('utf-8')
    tokenizer=ByteBPETokenizer.train(training,vocab_size=vocab_size)
    train_ids=torch.tensor(tokenizer.encode(training),dtype=torch.long)
    val_ids=torch.tensor(tokenizer.encode(validation),dtype=torch.long)
    return digest,tokenizer,train_ids,val_ids


@torch.no_grad()
def validation_loss(model,tokens,context,device,windows=8):
    model.eval()
    n=len(tokens)-context
    if n<=0:raise ValueError('Partición de validación insuficiente')
    offsets=torch.linspace(0,n-1,steps=min(windows,n)).long().tolist()
    losses=[]
    for start in offsets:
        x=tokens[start:start+context].unsqueeze(0).to(device)
        y=tokens[start+1:start+context+1].unsqueeze(0).to(device)
        _,loss=model(x,y);losses.append(loss.item())
    return sum(losses)/len(losses)


def safe_save(value,destination):
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    try:
        torch.save(value,tmp);os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def run(args):
    if min(args.steps,args.batch_size,args.grad_accum,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError('steps, batch-size, grad-accum, eval-every, save-every y cpu-threads >=1')
    if args.lr<=0:raise ValueError('lr debe ser positivo')
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    digest,tok,training,validation=prepare_data(args.data,args.seed,args.vocab_size)
    cfg=ModelConfig(vocab_size=tok.vocab_size,context=args.context,d_model=args.d_model,
                    layers=args.layers,heads=args.heads,dropout=args.dropout)
    if min(len(training),len(validation))<=cfg.context:
        raise ValueError('Dataset insuficiente: reducí --context o ampliá corpus')
    model=ByteTransformer(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    first=0;history=[]
    settings={'lr':args.lr,'batch_size':args.batch_size,'grad_accum':args.grad_accum}
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved.get('version')!=3:raise ValueError('Se requiere checkpoint MJON IA v0.3')
        if saved['dataset_sha256']!=digest or saved['seed']!=args.seed:
            raise ValueError('Cambió el dataset o la semilla; no se puede reanudar')
        if saved['model_config']!=cfg.to_dict() or saved['tokenizer']!=tok.to_dict():
            raise ValueError('Modelo/tokenizer incompatibles con checkpoint')
        if saved['training_settings']!=settings:
            raise ValueError('No cambiar lr/batch/grad-accum durante reanudación')
        first=int(saved['step'])
        if args.steps<=first:raise ValueError('--steps debe superar el paso guardado')
        history=list(saved['history'])
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'])
        if device=='cuda':
            if 'cuda_rng_states' not in saved:raise ValueError('Checkpoint sin RNG CUDA')
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        elif device!='cpu':
            raise ValueError('Reanudación exacta verificada solo para CPU y CUDA')
    out=Path(args.out)
    count=sum(p.numel() for p in model.parameters())
    print(f'MJON IA v0.3 | {count:,} parámetros | vocab {cfg.vocab_size} | train {len(training)} tokens | valid {len(validation)} | {device}')

    def checkpoint(step):
        data={'version':3,'model_config':cfg.to_dict(),'tokenizer':tok.to_dict(),
              'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
              'torch_rng_state':torch.get_rng_state(),'dataset_sha256':digest,
              'seed':args.seed,'step':step,'history':history,'training_settings':settings}
        if device=='cuda':data['cuda_rng_states']=torch.cuda.get_rng_state_all()
        safe_save(data,out)
        out.with_suffix('.metrics.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

    for step in range(first+1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);mean=0.0
        for _ in range(args.grad_accum):
            x,y=random_batch(training,args.batch_size,cfg.context,device)
            _,loss=model(x,y)
            (loss/args.grad_accum).backward()
            mean+=loss.item()/args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            val=validation_loss(model,validation,cfg.context,device)
            metrics={'step':step,'train_loss':round(mean,5),
                     'val_loss':round(val,5),
                     'val_perplexity':round(float(torch.exp(torch.tensor(min(val,20.0)))),2)}
            history.append(metrics);print('Métricas:',metrics)
        if step%args.save_every==0 or step==args.steps:checkpoint(step)
    return out


def parse_args():
    p=argparse.ArgumentParser(description='MJON IA v0.3 · entrenamiento independiente')
    p.add_argument('--data',default='data/synthetic_es.txt')
    p.add_argument('--out',default='checkpoints/mjon-v03.pt')
    p.add_argument('--resume',default=None)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--vocab-size',type=int,default=512)
    p.add_argument('--d-model',type=int,default=96)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--grad-accum',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0006)
    p.add_argument('--dropout',type=float,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    return p.parse_args()


if __name__=='__main__':
    run(parse_args())
