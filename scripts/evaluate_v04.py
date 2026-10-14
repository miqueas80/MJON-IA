"""MJON Bench: evaluación de generalización numérica en prompts no entrenados."""
import argparse
import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer


@torch.no_grad()
def score_completion(model, codec, prompt, answer):
    ids=codec.encode(prompt)
    target=codec.encode(answer)
    if not ids or not target:raise ValueError("Ejemplo de benchmark vacío")
    total=0.0
    for next_token in target:
        logits,_=model(torch.tensor([ids[-model.config.context:]],dtype=torch.long))
        total+=F.log_softmax(logits[0,-1],dim=-1)[next_token].item()
        ids.append(next_token)
    return total


@torch.no_grad()
def greedy(model,codec,prompt,count=10):
    ids=codec.encode(prompt);out=[]
    for _ in range(count):
        logits,_=model(torch.tensor([ids[-model.config.context:]],dtype=torch.long))
        pick=logits[0,-1].argmax().item()
        ids.append(pick);out.append(pick)
        if '\n' in codec.decode(out):break
    return codec.decode(out).split('\n')[0].strip()


def evaluate(checkpoint,bench,limit=12):
    torch.set_num_threads(2)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not state.get('tokenizer'):raise ValueError("Se requiere tokenizador BPE")
    codec=ByteBPETokenizer.from_dict(state['tokenizer'])
    model=ByteTransformer(ModelConfig(**state['model_config'])).eval()
    model.load_state_dict(state['state_dict'])
    data=json.loads(Path(bench).read_text(encoding='utf-8'))
    if data.get('version')!=1:raise ValueError('Benchmark desconocido')
    records=[]
    for item in data['probes']:
        if item['task']=='definition':continue
        if len(records)>=limit:break
        question,expected=item['question'],item['expected']
        if not re.fullmatch(r'\d+',expected):raise ValueError('Respuesta inválida')
        prompt=f"Usuario: {question}\nAsistente: "
        generated=greedy(model,codec,prompt)
        match=re.search(r'\d+',generated)
        correct=(match is not None and match.group(0)==expected)
        # Discriminación: puntuación del objetivo frente al entero contiguo.
        preferred=score_completion(model,codec,prompt,expected)>score_completion(model,codec,prompt,str(int(expected)+1))
        records.append({'task':item['task'],'question':question,'expected':expected,
                        'generated':generated[:120],'greedy_correct':correct,'pairwise_correct':preferred})
    if not records:raise ValueError('No existen ejemplos numéricos')
    return {'model':Path(checkpoint).stem,'steps':state['step'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'evaluated_numeric_probes':len(records),
            'greedy_exact_count':sum(x['greedy_correct'] for x in records),
            'pairwise_count':sum(x['pairwise_correct'] for x in records),
            'limitations':'Datos sintéticos y preguntas fuera de plantilla. No mide comprensión general ni conversación fiable.',
            'records':records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',default='data/benchmark_v04.json')
    p.add_argument('--checkpoints',nargs='+',default=['checkpoints/web/v03.pt','checkpoints/web/v04.pt'])
    p.add_argument('--out',default='docs/web/benchmarks.json')
    p.add_argument('--limit',type=int,default=12)
    args=p.parse_args()
    if not 1<=args.limit<=100:raise ValueError('limit inválido')
    results=[evaluate(path,args.benchmark,args.limit) for path in args.checkpoints]
    output={'schema':'mjonia.benchmark.v1','results':results}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for item in results:
        print(f"{item['model']}: exactas={item['greedy_exact_count']}/{item['evaluated_numeric_probes']} | preferencia={item['pairwise_count']}/{item['evaluated_numeric_probes']}")
    return output


if __name__=='__main__':main()
