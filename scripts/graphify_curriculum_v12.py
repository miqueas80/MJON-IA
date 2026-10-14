"""Convierte el grafo REAL de Graphify en ejemplos que entrenan MJON Original.

Solo consume relaciones EXTRACTED de graphify-out/graph.json. No introduce
Graphify, RAG, consultas de grafo ni motores externos al navegador.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

RELATIONS={"calls":"llama","imports":"importa","inherits":"hereda de",
           "contains":"contiene","method":"tiene método"}
TEMPLATES=(
    "Grafo MJON: ¿{a} {verb} {b}?",
    "Según el mapa de código, ¿{a} {verb} {b}?",
    "En el grafo de MJON, ¿está registrada la relación {relation} entre {a} y {b}?",
)


def names(node):
    label=node.get("label")
    path=node.get("source_file")
    if not isinstance(label,str) or not 2<=len(label)<=36:
        return None
    if node.get("external") or node.get("type")=="external":
        return None
    stem=Path(path).stem if isinstance(path,str) and path else ""
    if not stem or len(stem)>24:
        return None
    return f"{stem}.{label}"


def extract_edges(data):
    if not isinstance(data,dict) or not isinstance(data.get("nodes"),list) or not isinstance(data.get("links"),list):
        raise ValueError("Formato Graphify graph.json inválido")
    node_names={}
    for node in data["nodes"]:
        if isinstance(node,dict) and isinstance(node.get("id"),str):
            name=names(node)
            if name:node_names[node["id"]]=name
    candidate=[]
    for edge in data["links"]:
        if not isinstance(edge,dict) or edge.get("confidence")!="EXTRACTED":
            continue
        relation=edge.get("relation")
        if relation not in RELATIONS:
            continue
        source=node_names.get(edge.get("source"))
        target=node_names.get(edge.get("target"))
        if not source or not target or source==target or len(source)+len(target)>90:
            continue
        candidate.append((source,relation,target))
    # Ambigüedad: un mismo par de etiquetas puede designar entidades distintas.
    # Se aceptan triples exactamente equivalentes una única vez.
    return sorted(set(candidate))


def make_examples(edges,seed=1212):
    if len(edges)<12:
        raise ValueError(f"Graphify produjo pocas relaciones explícitas compatibles: {len(edges)}")
    rng=random.Random(seed)
    groups={}
    for item in edges:groups.setdefault(item[1],[]).append(item)
    complete=set(edges)
    pairs=[]
    for positive in edges:
        a,relation,b=positive
        candidates=sorted({target for _,rel,target in groups[relation]
                           if target!=b and (a,relation,target) not in complete})
        if not candidates:
            continue
        neg=(a,relation,rng.choice(candidates))
        pairs.append((positive,neg))
    if len(pairs)<8:raise ValueError("No hay suficientes ejemplos con contrastes verificables")
    rng.shuffle(pairs)
    n=len(pairs)
    # split por arista positiva (no la respuesta): conserva una fuente de verdad.
    ntest=max(2,int(n*.15))
    ndev=max(2,int(n*.15))
    if n-ntest-ndev<4:raise ValueError("Separación insuficiente")
    partitions={"train":pairs[:n-ntest-ndev],
                "dev":pairs[n-ntest-ndev:n-ntest],
                "test":pairs[n-ntest:]}
    generated={}
    for split,selected in partitions.items():
        rows=[]
        for pos,neg in selected:
            for triple,label in ((pos,"Sí."),(neg,"No.")):
                a,relation,b=triple
                templates=TEMPLATES[:2] if split=="train" else TEMPLATES[2:]
                for phrasing,template in enumerate(templates):
                    rows.append({"question":template.format(
                        a=a,b=b,relation=relation,verb=RELATIONS[relation]),
                        "answer":label,"subject":"codigo",
                        "relation":relation,"source":a,"target":b,
                        "edge_status":"EXTRACTED" if label=="Sí." else "NOT_RECORDED",
                        "positive_anchor":"|".join(pos),
                        "phrasing":phrasing})
        rng.shuffle(rows)
        generated[split]=rows
    # El negativo no puede corresponder a una arista real del grafo.
    assert all((x["source"],x["relation"],x["target"]) not in complete
               for part in generated.values() for x in part if x["answer"]=="No.")
    anchors=[set(r["positive_anchor"] for r in generated[k]) for k in ("train","dev","test")]
    assert not (anchors[0]&anchors[1] or anchors[0]&anchors[2] or anchors[1]&anchors[2])
    return generated


def run(source="graphify-out/v12/graph.json",out="data/graphify_v12",seed=1212):
    content=Path(source).read_bytes()
    data=json.loads(content)
    edges=extract_edges(data)
    generated=make_examples(edges,seed=seed)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    for part,rows in generated.items():
        (folder/(part+".json")).write_text(json.dumps(rows,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={
        "schema":"mjonia.graphify.v1","source":"Graphify-Labs/graphify",
        "source_version":"graphifyy 0.9.90",
        "graphify_license":"Apache-2.0","source_graph_sha256":hashlib.sha256(content).hexdigest(),
        "confidence_filter":"EXTRACTED only","closed_world_negatives":"NOT_RECORDED, not false in real code",
        "counts":{k:len(v) for k,v in generated.items()},
        "positive_relations":dict(Counter(r["relation"] for r in generated["train"]
                                         if r["answer"]=="Sí.")),
        "seed":seed,
        "note":"Solo para actualizar pesos neuronales durante entrenamiento; no se consulta el grafo en inferencia."
    }
    (folder/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MJON Graphify aprendizaje:",manifest["counts"],"relaciones extraídas",len(edges),flush=True)
    return generated,manifest


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="graphify-out/v12/graph.json")
    p.add_argument("--out",default="data/graphify_v12")
    p.add_argument("--seed",type=int,default=1212)
    a=p.parse_args()
    run(a.source,a.out,a.seed)
