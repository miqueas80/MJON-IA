"""MJON IA v0.6: ajuste supervisado de memoria de turnos y correcciones.

Usa los pesos de v0.5 y mezcla ejemplos de extracción simple para evitar
olvidar por completo el objetivo anterior. No usa APIs externas.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .finetune_v05 import collate, encode_record, evaluate
from .model import ByteTransformer, ModelConfig
from .tokenizer import ByteBPETokenizer
from .train_v03 import safe_save


def read_examples(path):
    raw = Path(path).read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("El corpus debe ser una lista no vacía")
    for row in records:
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("Formato de corpus inválido")
    return hashlib.sha256(raw).hexdigest(), records


def run(args):
    if min(args.steps, args.batch_size, args.eval_every, args.save_every, args.cpu_threads) < 1:
        raise ValueError("Pasos, lotes y núcleos deben ser positivos")
    if args.lr <= 0 or not 0 <= args.replay_fraction <= 1:
        raise ValueError("lr o replay_fraction inválido")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    base_bytes = Path(args.base).read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    if base.get("version") != 5 or not base.get("tokenizer"):
        raise ValueError("La base de MJON IA v0.6 debe ser un checkpoint SFT v0.5")
    cfg = ModelConfig(**base["model_config"])
    codec = ByteBPETokenizer.from_dict(base["tokenizer"])
    model = ByteTransformer(cfg)
    model.load_state_dict(base["state_dict"])

    train_sha, rows = read_examples(args.train)
    holdout_sha, reserved = read_examples(args.holdout)
    replay_sha, replay = read_examples(args.replay)
    if set(r["question"] for r in rows) & set(r["question"] for r in reserved):
        raise ValueError("Ejemplos repetidos en entrenamiento y holdout")
    if not any(r.get("case") == "correction" for r in rows + reserved):
        raise ValueError("Faltan pruebas de corrección")
    train = [encode_record(codec, r["question"], r["answer"], cfg.context) for r in rows]
    validation = [encode_record(codec, r["question"], r["answer"], cfg.context) for r in reserved]
    replay_data = [encode_record(codec, r["question"], r["answer"], cfg.context) for r in replay]
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    history = []
    start = 0
    signatures = {
        "base_sha256": base_hash, "train_sha256": train_sha,
        "holdout_sha256": holdout_sha, "replay_sha256": replay_sha,
        "learning_rate": args.lr, "replay_fraction": args.replay_fraction,
        "batch_size": args.batch_size, "seed": args.seed,
    }
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous.get("version") != 6 or any(previous.get(k) != v for k, v in signatures.items()):
            raise ValueError("Checkpoint incompatible con datos/configuración")
        if previous["model_config"] != cfg.to_dict() or previous["tokenizer"] != codec.to_dict():
            raise ValueError("Modelo/tokenizador incompatible")
        model.load_state_dict(previous["state_dict"])
        optim.load_state_dict(previous["optimizer_state_dict"])
        torch.set_rng_state(previous["torch_rng_state"])
        start = int(previous["step"])
        history = list(previous["history"])
        if start >= args.steps:
            raise ValueError("steps debe superar el paso guardado")
    output = Path(args.out)
    print(f"MJON Memory v0.6: {len(train)} entrenamiento, {len(reserved)} reservados, "
          f"{len(replay_data)} ejemplos de repaso", flush=True)
    for step in range(start + 1, args.steps + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        chosen = []
        for _ in range(args.batch_size):
            source = replay_data if torch.rand(()).item() < args.replay_fraction else train
            chosen.append(source[int(torch.randint(len(source), (1,)).item())])
        x, y = collate(chosen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            entry = {"step": step, "train_loss": round(loss.item(), 5),
                     "holdout_loss": evaluate(model, validation)}
            history.append(entry)
            print("Métricas:", entry, flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = {
                **signatures, "version": 6, "model_config": cfg.to_dict(),
                "tokenizer": codec.to_dict(), "state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "step": step, "history": history,
            }
            safe_save(checkpoint, output)
            output.with_suffix(".metrics.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args():
    p = argparse.ArgumentParser(description="MJON IA v0.6: memoria de contexto")
    p.add_argument("--base", default="checkpoints/web/v05.pt")
    p.add_argument("--train", default="data/memory_v06/train.json")
    p.add_argument("--holdout", default="data/memory_v06/holdout.json")
    p.add_argument("--replay", default="data/instruction_v05/train.json")
    p.add_argument("--out", default="checkpoints/web/v06.pt")
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=.0002)
    p.add_argument("--replay-fraction", type=float, default=.20)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--save-every", type=int, default=150)
    p.add_argument("--seed", type=int, default=606)
    p.add_argument("--cpu-threads", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
