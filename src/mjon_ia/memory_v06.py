"""MJON IA v0.6 — memoria conversacional sintética sin recuperación externa.

Tarea experimental: leer un atributo dado en un turno previo y aceptar una
corrección posterior. Los nombres reservados nunca entran al ajuste SFT.
Material original CC0; no contiene conversaciones de usuarios reales.
"""
import argparse
import json
import random
from pathlib import Path

from .instruction_v05 import ATTRIBUTES, TRAIN_ITEMS, TEST_ITEMS


def make_question(kind, item, first, latest, pattern):
    if kind not in ATTRIBUTES or first not in ATTRIBUTES[kind] or latest not in ATTRIBUTES[kind]:
        raise ValueError("Atributo o valor fuera de catálogo")
    if pattern == "recall_a":
        question = (
            f"El {item} tiene {kind} {first}.\n"
            "Asistente: Anotado.\n"
            f"Usuario: ¿Cuál es el {kind} del {item}?"
        )
    elif pattern == "recall_b":
        question = (
            f"Ficha: {item}, {kind} {first}.\n"
            "Asistente: Entendido.\n"
            f"Usuario: Decime el {kind} de {item}."
        )
    elif pattern == "update_a":
        question = (
            f"El {item} tiene {kind} {first}.\n"
            "Asistente: Anotado.\n"
            f"Usuario: Corrección: {kind} {latest}.\n"
            "Asistente: Actualizado.\n"
            f"Usuario: ¿Cuál es el {kind} actual del {item}?"
        )
    elif pattern == "update_b":
        question = (
            f"Ficha: {item}, {kind} {first}.\n"
            "Asistente: Entendido.\n"
            f"Usuario: Ahora {item}: {kind} {latest}.\n"
            "Asistente: Recibido.\n"
            f"Usuario: Decime el {kind} de {item} ahora."
        )
    else:
        raise ValueError("Patrón desconocido")
    return question


def records_for(items, seed, max_records=None):
    rows = []
    for item in items:
        for kind, values in ATTRIBUTES.items():
            for first in values:
                for pattern in ("recall_a", "recall_b", "update_a", "update_b"):
                    # En correcciones, valor nuevo distinto al anterior.
                    latest = values[(values.index(first) + len(item) + len(kind)) % len(values)]
                    if latest == first:
                        latest = values[(values.index(first) + 1) % len(values)]
                    if pattern.startswith("recall"):
                        latest = first
                    q = make_question(kind, item, first, latest, pattern)
                    rows.append({
                        "question": q, "answer": latest, "task": kind,
                        "case": "correction" if pattern.startswith("update") else "recall",
                        "item": item, "pattern": pattern
                    })
                # El primer turno también debe poder contestarse en el chat.
                for prompt, response, pattern in (
                    (f"El {item} tiene {kind} {first}.", "Anotado.", "ack_initial"),
                    (f"Corrección del {item}: {kind} {first}.", "Actualizado.", "ack_update"),
                ):
                    rows.append({
                        "question": prompt, "answer": response, "task": kind,
                        "case": "ack", "item": item, "pattern": pattern
                    })
    random.Random(seed).shuffle(rows)
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records debe ser positivo")
        rows = rows[:max_records]
    return rows


def build(seed=606, limit=2500):
    training = records_for(TRAIN_ITEMS, seed, limit)
    reserved = records_for(TEST_ITEMS, seed + 100)
    train_questions = {r["question"] for r in training}
    test_questions = {r["question"] for r in reserved}
    if train_questions & test_questions:
        raise ValueError("Fuga de preguntas al conjunto reservado")
    if any(f" {name} " in q for name in TEST_ITEMS for q in train_questions):
        raise ValueError("Fuga de nombres de evaluación")
    return training, reserved


def save(out_dir="data/memory_v06", seed=606, limit=2500):
    train, test = build(seed, limit)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.json", train), ("holdout.json", test)):
        (destination / name).write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"MJON Memory: {len(train)} ejemplos de entrenamiento; "
          f"{len(test)} reservados, incluyendo correcciones", flush=True)
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/memory_v06")
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--limit", type=int, default=2500)
    a = parser.parse_args()
    save(a.out, a.seed, a.limit)
