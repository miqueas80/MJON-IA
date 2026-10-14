"""MJON Memory Bench: recuperación de un turno previo y correcciones posteriores."""
import argparse
import json
from pathlib import Path

import torch
from mjon_ia.model import ByteTransformer, ModelConfig
from mjon_ia.tokenizer import ByteBPETokenizer
from benchmark_v05 import normalize, predict, score as firstwords_score


def select_balanced(records, per_group=8):
    chosen = []
    for case in ("recall", "correction"):
        for kind in ("color", "material", "estado"):
            group = [r for r in records if r["case"] == case and r["task"] == kind]
            chosen.extend(group[:per_group])
    return chosen


def evaluate_checkpoint(checkpoint, rows):
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    codec = ByteBPETokenizer.from_dict(state["tokenizer"])
    model = ByteTransformer(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["state_dict"])
    data = []
    for row in rows:
        result = predict(model, codec, row["question"], max_tokens=16)
        data.append({
            "case": row["case"], "task": row["task"], "expected": row["answer"],
            "question": row["question"], "generated": result[:100],
            "exact": normalize(result) == normalize(row["answer"]),
        })
    groups = {}
    for case in ("recall", "correction"):
        relevant = [x for x in data if x["case"] == case]
        groups[case] = {"correct": sum(x["exact"] for x in relevant), "total": len(relevant)}
    return {
        "checkpoint": Path(checkpoint).stem, "total": len(data),
        "correct": sum(x["exact"] for x in data),
        "by_case": groups, "samples": data,
    }


def run(base, tuned, holdout, firstwords, output, per_group=8):
    data = json.loads(Path(holdout).read_text(encoding="utf-8"))
    selected = select_balanced(data, per_group)
    if len(selected) != per_group * 6:
        raise ValueError("Benchmark desbalanceado")
    baseline = evaluate_checkpoint(base, selected)
    trained = evaluate_checkpoint(tuned, selected)
    previous_firstwords = firstwords_score(base, firstwords, limit=48)
    current_firstwords = firstwords_score(tuned, firstwords, limit=48)
    result = {
        "schema": "mjonia.memory.v1", "models": [baseline, trained],
        "retention": {
            "before": {"correct": previous_firstwords["exact"], "total": previous_firstwords["total"]},
            "after": {"correct": current_firstwords["exact"], "total": current_firstwords["total"]}
        },
        "scope": "48 preguntas de recuerdos y correcciones en objetos excluidos del ajuste. "
                 "No mide memoria persistente ni conversación abierta."
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in result["models"]:
        print(f"MJON Memory {item['checkpoint']}: {item['correct']}/{item['total']} exactas; "
              f"recuerdo {item['by_case']['recall']['correct']}/{item['by_case']['recall']['total']}, "
              f"correcciones {item['by_case']['correction']['correct']}/{item['by_case']['correction']['total']}",
              flush=True)
    print("Retención primeras palabras:", result["retention"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="checkpoints/web/v05.pt")
    parser.add_argument("--tuned", default="checkpoints/web/v06.pt")
    parser.add_argument("--holdout", default="data/memory_v06/holdout.json")
    parser.add_argument("--firstwords", default="data/instruction_v05/holdout.json")
    parser.add_argument("--out", default="docs/web/memory.json")
    parser.add_argument("--per-group", type=int, default=8)
    a = parser.parse_args()
    run(a.base, a.tuned, a.holdout, a.firstwords, a.out, a.per_group)
