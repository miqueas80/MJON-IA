"""Datos instruccionales MJON para entrenar PESOS propios del Neural Core.

El conjunto es pequeño y original; sirve como arranque reproducible de SFT,
NO como sustituto del preentrenamiento con grandes corpus autorizados.
"""
import argparse
import json
from pathlib import Path

from mjon_ia.knowledge_v11 import build,parse


def build_corpus(destination="data/neural_corpus/knowledge_sft.jsonl"):
    path=Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    trained,paraphrases,heldout=build(seed=1111)
    # Los conceptos se agrupan antes de separar, incluso si hay paráfrasis.
    records=[]
    questions={}
    for row in trained+paraphrases+heldout:
        record={
            "instruction":row["question"],
            "response":row["answer"],
            "source_id":"MJON-IA/original-conceptos-v1.1",
            "group_id":row["concept"],
            "license":"CC0-1.0",
            "subject":row["subject"],
        }
        existing=questions.setdefault(record["instruction"],record["group_id"])
        if existing!=record["group_id"]:
            raise ValueError("La pregunta aparece en conceptos distintos")
        records.append(record)
    path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in records)+"\n",
                    encoding="utf-8")
    print("MJON Neural: corpus SFT propio",len(records),"preguntas",
          "120 conceptos con split posterior por concepto",flush=True)
    return records


def build_pretrain(destination="data/neural_corpus/knowledge_pretrain.jsonl"):
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for subject,concepts in parse().items():
        for i,(concept,definition) in enumerate(concepts):
            records.append({
                "text":f"Área: {subject}. Concepto: {concept}. Definición: {definition}. "
                       f"Pregunta: ¿Qué significa {concept}? Respuesta: {definition}",
                "source_id":"MJON-IA/original-conceptos-v1.1",
                "document_id":f"{subject}-{i}",
                "license":"CC0-1.0",
            })
    path.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in records)+"\n",
                    encoding="utf-8")
    print("MJON Neural: corpus pretrain de laboratorio",len(records),
          "documentos. No es un dataset de escala industrial.",flush=True)
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/neural_corpus")
    a=p.parse_args()
    root=Path(a.out)
    build_pretrain(root/"knowledge_pretrain.jsonl")
    build_corpus(root/"knowledge_sft.jsonl")
