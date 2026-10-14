"""MJON First Words: generación libre en ítems no vistos durante entrenamiento SFT."""
import argparse
import json
import re
from pathlib import Path
import torch
from mjon_ia.model import ByteTransformer,ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer

def normalize(text):
    return re.sub(r"\s+"," ",text.strip().lower()).rstrip(".!¡?¿ ").strip()

@torch.no_grad()
def predict(model,codec,question,max_tokens=16):
    model.eval()
    ids=codec.encode(f"Usuario: {question}\nAsistente:")
    out=[]
    for _ in range(max_tokens):
        logits,_=model(torch.tensor([ids[-model.config.context:]],dtype=torch.long))
        token=int(logits[0,-1].argmax().item())
        ids.append(token);out.append(token)
        answer=codec.decode(out)
        if "\n" in answer:return answer.split("\n",1)[0]
    return codec.decode(out)

def score(checkpoint,holdout,limit=48):
    torch.set_num_threads(2)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    codec=ByteBPETokenizer.from_dict(state["tokenizer"])
    model=ByteTransformer(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["state_dict"])
    records=json.loads(Path(holdout).read_text(encoding="utf-8"))
    pool={key:[] for key in ("color","material","estado")}
    for row in records:pool[row["task"]].append(row)
    selected=[]
    while len(selected)<min(limit,len(records)) and any(pool.values()):
        for key in pool:
            if pool[key] and len(selected)<limit:selected.append(pool[key].pop(0))
    samples=[]
    for row in selected:
        output=predict(model,codec,row["question"])
        samples.append({"task":row["task"],"question":row["question"],"expected":row["answer"],
                        "generated":output[:100],"exact":normalize(output)==normalize(row["answer"])})
    correct=sum(x["exact"] for x in samples)
    tasks={kind:{"correct":sum(r["exact"] for r in samples if r["task"]==kind),
                 "count":sum(r["task"]==kind for r in samples)} for kind in pool}
    return {"checkpoint":Path(checkpoint).stem,"total":len(samples),"exact":correct,
            "percentage":round(100*correct/max(1,len(samples)),2),"per_task":tasks,
            "samples":samples,"scope":"Extracción de atributos explícitos en objetos ajenos al ajuste SFT; no evalúa razonamiento general."}

def run(base,tuned,holdout,out,limit=48):
    results=[score(base,holdout,limit),score(tuned,holdout,limit)]
    report={"schema":"mjonia.firstwords.v1","models":results,
            "interpretation":"Pruebas de extracción literal de color, material y estado; no demuestran conversación abierta."}
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for item in results:
        print(f"{item['checkpoint']}: {item['exact']}/{item['total']} respuestas exactas ({item['percentage']}%)",flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="checkpoints/web/v04.pt")
    p.add_argument("--tuned",default="checkpoints/web/v05.pt")
    p.add_argument("--holdout",default="data/instruction_v05/holdout.json")
    p.add_argument("--out",default="docs/web/firstwords.json")
    p.add_argument("--limit",type=int,default=48)
    a=p.parse_args();run(a.base,a.tuned,a.holdout,a.out,a.limit)
