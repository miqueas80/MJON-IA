"""MJON Original v1.2: entrenamiento con relaciones EXTRACTED de Graphify.

Actualiza los PESOS del Transformer existente. Graphify SOLO actúa en build
para extraer relaciones. No se carga en navegador, no es otro cerebro.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .distill_v10 import choose_candidate, evaluate_development, frozen_distribution_loss
from .finetune_v05 import collate, encode_record, evaluate
from .finetune_v06 import read_examples
from .model import ByteTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save


def load_data(args):
    base=torch.load(args.base,map_location="cpu",weights_only=True)
    if base.get("version")!=11:raise ValueError("MJON Graphify debe comenzar desde v1.1")
    meta=json.loads(Path(args.graph_manifest).read_text(encoding="utf-8"))
    if meta.get("schema")!="mjonia.graphify.v1" or meta.get("confidence_filter")!="EXTRACTED only":
        raise ValueError("Metadatos Graphify insuficientes")
    rows={}
    for key,path in (("graph",args.graph),("graph_dev",args.graph_dev),
                     ("graph_test",args.graph_test),
                     ("knowledge",args.knowledge),("knowledge_dev",args.knowledge_dev),
                     ("school",args.school),("memory",args.memory),
                     ("school_holdout",args.school_holdout),
                     ("school_transfer",args.school_transfer),
                     ("memory_holdout",args.memory_holdout)):
        _,rows[key]=read_examples(path)
    if len(rows["graph"])<20 or len(rows["graph_dev"])<2 or len(rows["graph_test"])<2:
        raise ValueError("Graphify no proporcionó relaciones suficientes para aprender")
    all_edges=[]
    for key in ("graph","graph_dev","graph_test"):
        anchors={r["positive_anchor"] for r in rows[key]}
        all_edges.append(anchors)
    if any(all_edges[i] & all_edges[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("Fuga de relaciones entre entrenamiento y examen")
    if len(rows["knowledge"])<400 or len(rows["knowledge_dev"])<100:
        raise ValueError("Falta repaso curricular de v1.1")
    cfg=ModelConfig(**base["model_config"])
    codec=ByteBPETokenizer.from_dict(base["tokenizer"])
    pools={}
    for key in ("graph","graph_dev","knowledge","knowledge_dev","school","memory"):
        encoded=[]
        for row in rows[key]:
            try:encoded.append(encode_record(codec,row["question"],row["answer"],cfg.context))
            except ValueError:
                if key in ("knowledge","knowledge_dev"):raise
        pools[key]=encoded
        if not encoded:raise ValueError(f"Ningún ejemplo válido de {key}")
    # No convertir el examen Graphify en datos del optimizador.
    if len(pools["graph"])<16 or len(pools["graph_dev"])<2:
        raise ValueError("Demasiados enlaces Graphify superaron la ventana BPE")
    # Retención: la selección usa antiguas preguntas 10..13, no el examen 0..9.
    development=[]
    for subject in ("matematica","lengua","ciencias","logica"):
        development += [r for r in rows["school_holdout"] if r["subject"]==subject][10:14]
    for subject in ("matematica","lengua","ciencias","logica"):
        development += [r for r in rows["school_transfer"] if r["subject"]==subject][10:14]
    development += [r for r in rows["memory_holdout"] if r.get("case") in
                    ("recall","correction")][:8]
    if len(development)!=40:raise ValueError("Calibración de retención incompleta")
    seen={r["question"] for r in rows["graph"]+rows["knowledge"]+rows["school"]+rows["memory"]}
    if seen & {r["question"] for r in rows["graph_dev"]+rows["graph_test"]+development}:
        raise ValueError("Fuga de preguntas de examen")
    return base,codec,pools,development,meta


def run(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every,args.cpu_threads)<1:
        raise ValueError("Pasos y lotes inválidos")
    if not 0<args.graph_fraction<1 or not 0<args.knowledge_fraction<1-args.graph_fraction:
        raise ValueError("Fracciones incorrectas")
    if not 0<args.memory_fraction<1-args.graph_fraction-args.knowledge_fraction:
        raise ValueError("Memoria fuera de rango")
    if not 0<=args.kd_weight<=10 or args.lr<=0:raise ValueError("LR o KD inválidos")
    torch.manual_seed(args.seed);torch.set_num_threads(args.cpu_threads)
    base,codec,pools,development,meta=load_data(args)
    cfg=ModelConfig(**base["model_config"])
    student=ByteTransformer(cfg);student.load_state_dict(base["state_dict"])
    reference=ByteTransformer(cfg).eval();reference.load_state_dict(base["state_dict"])
    for param in reference.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=.01)
    signature={
        "base_sha256":hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "graph_sha256":hashlib.sha256(Path(args.graph).read_bytes()).hexdigest(),
        "graph_dev_sha256":hashlib.sha256(Path(args.graph_dev).read_bytes()).hexdigest(),
        "manifest_sha256":hashlib.sha256(Path(args.graph_manifest).read_bytes()).hexdigest(),
        "knowledge_sha256":hashlib.sha256(Path(args.knowledge).read_bytes()).hexdigest(),
        "knowledge_dev_sha256":hashlib.sha256(Path(args.knowledge_dev).read_bytes()).hexdigest(),
        "school_sha256":hashlib.sha256(Path(args.school).read_bytes()).hexdigest(),
        "memory_sha256":hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
        "seed":args.seed,"lr":args.lr,"batch_size":args.batch_size,
        "graph_fraction":args.graph_fraction,
        "knowledge_fraction":args.knowledge_fraction,
        "memory_fraction":args.memory_fraction,
        "kd_weight":args.kd_weight,"eval_every":args.eval_every,
    }
    original_dev=evaluate_development(reference,codec,development)
    original_graph_loss=evaluate(reference,pools["graph_dev"],count=len(pools["graph_dev"]))
    original_knowledge_loss=evaluate(reference,pools["knowledge_dev"],count=25)
    best_loss=original_graph_loss
    best_step=0
    start=0
    history=[]
    output=Path(args.out);last=output.with_name(output.stem+".last.pt")
    if args.resume:
        prev=torch.load(args.resume,map_location="cpu",weights_only=True)
        if prev.get("version")!=12 or any(prev.get(k)!=v for k,v in signature.items()):
            raise ValueError("Checkpoint Graphify incompatible con el entrenamiento")
        if prev["model_config"]!=base["model_config"] or prev["tokenizer"]!=base["tokenizer"]:
            raise ValueError("Arquitectura o tokenizador alterado")
        student.load_state_dict(prev["state_dict"])
        optimizer.load_state_dict(prev["optimizer_state_dict"])
        torch.set_rng_state(prev["torch_rng_state"])
        start=int(prev["step"]);history=list(prev["history"])
        best_loss=float(prev["best_graph_loss"]);best_step=int(prev["best_step"])
        if args.steps<=start:raise ValueError("steps debe aumentar para reanudar")
    else:
        history=[{"step":0,"graph_dev_loss":original_graph_loss,
                  "knowledge_dev_loss":original_knowledge_loss,
                  "development":original_dev,"promoted":True}]
    def snapshot(step):
        return {"version":12,**signature,
                "model_config":base["model_config"],"tokenizer":base["tokenizer"],
                "state_dict":student.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                "torch_rng_state":torch.get_rng_state(),"step":step,"history":history,
                "best_step":best_step,"best_graph_loss":best_loss,
                "baseline_graph_loss":original_graph_loss,
                "baseline_knowledge_loss":original_knowledge_loss,
                "baseline_development":original_dev,
                "graphify_source":meta["source"],"graphify_license":meta["graphify_license"],
                "graphify_graph_sha256":meta["source_graph_sha256"]}
    if not args.resume:safe_save(snapshot(0),output)
    print(f"MJON Graphify v1.2: {len(pools['graph'])} ejemplos de aristas "
          f"de código, {len(pools['graph_dev'])} desarrollo, "
          f"calibración base {original_dev}, graph loss {original_graph_loss}",flush=True)
    for step in range(start+1,args.steps+1):
        student.train();optimizer.zero_grad(set_to_none=True)
        selected=[]
        for _ in range(args.batch_size):
            value=float(torch.rand(()))
            group=("graph" if value<args.graph_fraction else
                   "knowledge" if value<args.graph_fraction+args.knowledge_fraction else
                   "memory" if value<args.graph_fraction+args.knowledge_fraction+args.memory_fraction
                   else "school")
            dataset=pools[group]
            ix=int(torch.randint(len(dataset),(1,)))
            selected.append((group,dataset[ix]))
        x,y=collate([row for _,row in selected])
        logits,_=student(x)
        sup=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                            y.reshape(-1),ignore_index=-100)
        replay=[i for i,(grp,_) in enumerate(selected) if grp!="graph"]
        kd=logits.sum()*0
        if replay and args.kd_weight>0:
            with torch.no_grad():old_logits,_=reference(x[replay])
            kd=frozen_distribution_loss(logits[replay],old_logits,y[replay])
        loss=sup+args.kd_weight*kd
        loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            graph_loss=evaluate(student,pools["graph_dev"],count=len(pools["graph_dev"]))
            knowledge_loss=evaluate(student,pools["knowledge_dev"],count=25)
            dev=evaluate_development(student,codec,development)
            retained=choose_candidate(original_dev,dev,original_graph_loss,graph_loss)
            retained=retained and knowledge_loss<=original_knowledge_loss+.12
            promote=retained and graph_loss<best_loss-.00001
            if promote:best_loss=graph_loss;best_step=step
            stats={"step":step,"train_loss":round(float(sup.item()),5),
                   "kd_loss":round(float(kd.item()),5),
                   "graph_dev_loss":graph_loss,
                   "knowledge_dev_loss":knowledge_loss,"development":dev,
                   "eligible":retained,"promoted":promote}
            history.append(stats)
            if promote:safe_save(snapshot(step),output)
            print("MJON Graphify:",stats,flush=True)
        if step%args.save_every==0 or step==args.steps:
            safe_save(snapshot(step),last)
    best=torch.load(output,map_location="cpu",weights_only=True)
    output.with_suffix(".metrics.json").write_text(json.dumps(
        {"selected_step":best["step"],"base_dev":original_dev,
         "base_graph_loss":original_graph_loss,
         "history":history},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MJON Graphify v1.2: mejor checkpoint",best["step"],
          "de",args.steps,flush=True)
    return output


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="checkpoints/web/v11.pt")
    p.add_argument("--graph-manifest",default="data/graphify_v12/manifest.json")
    p.add_argument("--graph",default="data/graphify_v12/train.json")
    p.add_argument("--graph-dev",default="data/graphify_v12/dev.json")
    p.add_argument("--graph-test",default="data/graphify_v12/test.json")
    p.add_argument("--knowledge",default="data/knowledge_v11/train.json")
    p.add_argument("--knowledge-dev",default="data/knowledge_v11/paraphrase.json")
    p.add_argument("--school",default="data/school_v07/train.json")
    p.add_argument("--memory",default="data/memory_v06/train.json")
    p.add_argument("--school-holdout",default="data/school_v07/holdout.json")
    p.add_argument("--school-transfer",default="data/school_v07/transfer.json")
    p.add_argument("--memory-holdout",default="data/memory_v06/holdout.json")
    p.add_argument("--out",default="checkpoints/web/v12.pt")
    p.add_argument("--resume",default=None)
    p.add_argument("--steps",type=int,default=360)
    p.add_argument("--batch-size",type=int,default=6)
    p.add_argument("--graph-fraction",type=float,default=.35)
    p.add_argument("--knowledge-fraction",type=float,default=.35)
    p.add_argument("--memory-fraction",type=float,default=.12)
    p.add_argument("--kd-weight",type=float,default=.55)
    p.add_argument("--lr",type=float,default=.00006)
    p.add_argument("--eval-every",type=int,default=60)
    p.add_argument("--save-every",type=int,default=60)
    p.add_argument("--seed",type=int,default=1212)
    p.add_argument("--cpu-threads",type=int,default=2)
    return p.parse_args()


if __name__=="__main__":run(parse_args())
