"""MJON IA v0.7: cuatro materias, ejercicios originales CC0 y evaluación separada.

Demostración curricular, NO contenido de secundaria completo.
Los ejemplos se crean con reglas transparentes. Las pruebas de generalización
incluyen reformulaciones no usadas para entrenamiento.
"""
import argparse
import json
import random
from pathlib import Path

TRAIN_NAMES = ("atlas","banco","lápiz","cuaderno","péndulo","reloj","vaso","pincel",
               "termómetro","cilindro","frasco","telar","motor","foco","imán","cable")
TEST_NAMES = ("sextante","periscopio","microscopio","catalejo","anemómetro","giróscopo")
COLORS = ("azul","rojo","verde","negro","blanco","violeta","amarillo","naranja")
MEASURES = ("temperatura","masa","volumen","longitud","tiempo","velocidad")
MATERIALS = ("metal","vidrio","papel","madera","plástico","cerámica")
ANIMALS = ("zorro","gato","perro","caballo","puma","lobo","ratón","águila")
TEST_ANIMALS = ("cisne","delfín","jaguar","ballena","tortuga","hormiga")


def exercise(subject, seed, index, mode):
    """mode=train, holdout o transfer (formulación distinta)."""
    rng = random.Random((seed + 47) * 1000003 + index * 11939)
    names = TRAIN_NAMES if mode == "train" else TEST_NAMES
    a, b = rng.choice(names), rng.choice(names)
    if a == b:
        b = names[(names.index(a) + 1) % len(names)]
    if subject == "matematica":
        start = rng.randint(3, 24) if mode == "train" else rng.randint(25, 59)
        added, removed = rng.randint(2, 9), rng.randint(1, 5)
        if mode == "transfer":
            prompt = f"Calculá la cantidad final si hay {start}, se suman {added} y se quitan {removed}."
        elif index % 2:
            prompt = f"En una caja hay {start} fichas. Entran {added} fichas y salen {removed}. ¿Cuántas hay?"
        else:
            prompt = f"Un grupo guarda {start} cartas, recibe {added} y pierde {removed}. ¿Cuántas cartas quedan?"
        answer = str(start + added - removed)
    elif subject == "lengua":
        first, second = rng.sample(COLORS, 2)
        if mode == "transfer":
            prompt = f"Según el texto, {a} es {first} y {b} es {second}. ¿Qué color corresponde a {a}?"
        elif index % 2:
            prompt = f"Texto: el objeto {a} tiene color {first}. El objeto {b} es {second}. ¿De qué color es {a}?"
        else:
            prompt = f"Lectura: {a} figura como {first}; {b} como {second}. Respondé el color de {a}."
        answer = first
    elif subject == "ciencias":
        measure, other = rng.sample(MEASURES, 2)
        if mode == "transfer":
            prompt = f"Apunte científico: {a} mide {measure}, mientras {b} mide {other}. ¿Qué magnitud mide {b}?"
            answer = other
        elif index % 2:
            prompt = f"Ficha experimental: el instrumento {a} mide {measure}. El {b} mide {other}. ¿Qué mide {a}?"
            answer = measure
        else:
            prompt = f"Laboratorio: {a} sirve para medir {measure} y {b} para medir {other}. Indicá qué mide {a}."
            answer = measure
    elif subject == "logica":
        first, second, third = rng.sample(ANIMALS if mode == "train" else TEST_ANIMALS, 3)
        if mode == "transfer":
            prompt = f"Orden de llegada: {first} precede a {second} y {second} precede a {third}. ¿Quién llegó último?"
            answer = third
        elif index % 2:
            prompt = f"Si {first} llegó antes que {second} y {second} llegó antes que {third}, ¿quién llegó primero?"
            answer = first
        else:
            prompt = f"En una fila, {first} está antes que {second}, y {second} antes que {third}. ¿Quién es el primero?"
            answer = first
    else:
        raise ValueError("Materia desconocida")
    return {"question": prompt, "answer": answer, "subject": subject,
            "mode": mode}


SUBJECTS = ("matematica","lengua","ciencias","logica")


def generate(seed=707, per_subject=1200, heldout_per_subject=32):
    if per_subject < 10 or heldout_per_subject < 4:
        raise ValueError("Muy pocos ejercicios")
    training, unseen, transfer = [], [], []
    for j, subject in enumerate(SUBJECTS):
        for n in range(per_subject):
            training.append(exercise(subject, seed, j * 200000 + n, "train"))
        for n in range(heldout_per_subject):
            unseen.append(exercise(subject, seed + 1000, j * 200000 + n, "holdout"))
            transfer.append(exercise(subject, seed + 2000, j * 200000 + n, "transfer"))
    rng = random.Random(seed)
    rng.shuffle(training);rng.shuffle(unseen);rng.shuffle(transfer)
    questions = [set(row["question"] for row in rows) for rows in (training,unseen,transfer)]
    if questions[0] & questions[1] or questions[0] & questions[2]:
        raise ValueError("Contaminación de preguntas")
    # Validación estricta para Lengua y Ciencias: nuevos nombres fuera de train.
    for row in training:
        if row["subject"] in ("lengua","ciencias") and any(name in row["question"] for name in TEST_NAMES):
            raise ValueError("Contaminación de entidades")
    return training, unseen, transfer


def save(out="data/school_v07",seed=707,per_subject=1200,heldout=32):
    train,unseen,transfer=generate(seed,per_subject,heldout)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    for name,rows in (("train",train),("holdout",unseen),("transfer",transfer)):
        (folder/f"{name}.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"MJON Escuela: {len(train)} ejercicios SFT; {len(unseen)} reservados; "
          f"{len(transfer)} con nuevas formulaciones.",flush=True)
    return train,unseen,transfer


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/school_v07")
    p.add_argument("--seed",type=int,default=707)
    p.add_argument("--per-subject",type=int,default=1200)
    p.add_argument("--heldout",type=int,default=32)
    a=p.parse_args();save(a.out,a.seed,a.per_subject,a.heldout)
