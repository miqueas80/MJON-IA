"""Prompts originales v0.8 para transferencia de conocimiento (MIT).

El profesor Qwen no forma parte del modelo MJON. Genera pares pregunta/respuesta
y se conservan procedencia, revisiones, filtros y pruebas independientes.
"""
import hashlib
import json
import random
from pathlib import Path

TOPICS = {
    "biologia": [
        ("¿Qué es una célula?", "Explicá qué estudia la biología celular."),
        ("¿Qué función cumple el ADN?", "¿Para qué sirve el ADN en los seres vivos?"),
        ("¿Qué es la fotosíntesis?", "¿Cómo obtienen energía las plantas con luz?"),
        ("¿Qué es un ecosistema?", "¿Cómo interactúan seres vivos y ambiente?"),
        ("¿Qué es una especie biológica?", "¿Qué significa especie en biología?"),
        ("¿Qué hacen los glóbulos rojos?", "¿Para qué sirven los eritrocitos?"),
        ("¿Qué es la mitosis?", "¿Para qué ocurre la división mitótica?"),
        ("¿Qué es una neurona?", "¿Qué función realiza una neurona?"),
    ],
    "quimica": [
        ("¿Qué es un átomo?", "Definí la unidad llamada átomo."),
        ("¿Qué es una molécula?", "Explicá el concepto de molécula."),
        ("¿Qué diferencia hay entre un elemento y un compuesto?", "¿En qué se distingue un elemento químico de un compuesto?"),
        ("¿Qué representa el pH?", "¿Para qué se utiliza una escala de pH?"),
        ("¿Qué es una reacción química?", "¿Qué ocurre cuando una sustancia reacciona químicamente?"),
        ("¿Qué es un enlace iónico?", "Definí el enlace entre iones."),
        ("¿Qué es una disolución?", "¿Qué caracteriza a una solución química?"),
        ("¿Qué es la tabla periódica?", "¿Cómo se organizan los elementos químicos?"),
    ],
    "fisica": [
        ("¿Qué es la velocidad?", "¿Cómo se define la velocidad en física?"),
        ("¿Qué es una fuerza?", "¿A qué se llama fuerza en física?"),
        ("¿Qué indica la temperatura?", "¿Qué magnitud mide un termómetro?"),
        ("¿Qué es la energía cinética?", "Definí la energía del movimiento."),
        ("¿Qué es la densidad?", "¿Qué significa la densidad de un material?"),
        ("¿Qué es la gravedad?", "¿Cómo se describe la gravedad?"),
        ("¿Qué es la corriente eléctrica?", "¿Qué representa la intensidad de corriente eléctrica?"),
        ("¿Qué es una onda sonora?", "¿Cómo se propaga el sonido?"),
    ],
    "matematica": [
        ("¿Qué es una fracción?", "¿Qué representa una fracción matemática?"),
        ("¿Qué significa porcentaje?", "¿Cómo interpretar un porcentaje?"),
        ("¿Qué es una ecuación?", "¿A qué se llama ecuación matemática?"),
        ("¿Qué es un número primo?", "Definí un número primo."),
        ("¿Qué es un triángulo rectángulo?", "¿Qué caracteriza al triángulo con ángulo recto?"),
        ("¿Qué significa el valor absoluto?", "¿Qué representa el módulo de un número real?"),
        ("¿Qué es una variable matemática?", "¿Qué representa una variable en álgebra?"),
        ("¿Qué es una función?", "¿Cómo definir una función matemática?"),
    ],
    "lengua": [
        ("¿Qué es un sustantivo?", "¿Para qué se usa un sustantivo en una oración?"),
        ("¿Qué es un verbo?", "¿Qué expresa un verbo?"),
        ("¿Qué es un sinónimo?", "¿Qué significa que dos palabras sean sinónimas?"),
        ("¿Qué es un antónimo?", "¿Cómo se llama una palabra de sentido opuesto?"),
        ("¿Qué es un párrafo?", "¿Cuál es la función de un párrafo?"),
        ("¿Qué es la idea principal de un texto?", "¿Cómo reconocer la idea central de una lectura?"),
        ("¿Qué es una metáfora?", "Explicá qué es una metáfora."),
        ("¿Qué es un adjetivo?", "¿Qué aporta un adjetivo a una oración?"),
    ],
    "informatica": [
        ("¿Qué es un algoritmo?", "Definí una secuencia de instrucciones llamada algoritmo."),
        ("¿Qué es una variable en programación?", "¿Para qué sirve una variable en código?"),
        ("¿Qué es una función en programación?", "¿Cómo se usa una función programada?"),
        ("¿Qué significa código abierto?", "¿Qué permite el software de código abierto?"),
        ("¿Qué es la memoria RAM?", "¿Qué función cumple la RAM en una computadora?"),
        ("¿Qué es una base de datos?", "Explicá para qué se usa una base de datos."),
        ("¿Qué es una red local?", "¿Cómo se describe una red de computadoras local?"),
        ("¿Qué es un sistema operativo?", "¿Qué función cumple el sistema operativo?"),
    ],
}


def build_prompts(seed=808, train_limit=42, holdout_limit=12):
    """Partición por concepto: nunca se entrena sobre el concepto evaluado."""
    if not 1 <= train_limit <= 42 or not 1 <= holdout_limit <= 12:
        raise ValueError("Límites: train 1..42, holdout 1..12")
    training, holdout = [], []
    for topic, pairs in TOPICS.items():
        for index, (train, test) in enumerate(pairs):
            # Seis temas reservados (1 por asignatura), no solo preguntas repetidas.
            if index == 7:
                holdout.append({"question": test, "subject": topic, "concept": f"{topic}:{index}"})
            else:
                training.append({"question": train, "subject": topic, "concept": f"{topic}:{index}"})
    random.Random(seed).shuffle(training)
    random.Random(seed + 1).shuffle(holdout)
    return training[:train_limit], holdout[:holdout_limit]


def verify_separation(training, heldout):
    train_ids = {x["concept"] for x in training}
    heldout_ids = {x["concept"] for x in heldout}
    train_questions = {x["question"] for x in training}
    heldout_questions = {x["question"] for x in heldout}
    if train_ids & heldout_ids or train_questions & heldout_questions:
        raise ValueError("Fuga del benchmark al entrenamiento")
    return True


def manifest(model_name, revision, rows, holdout):
    verify_separation(rows, holdout)
    return {
        "schema": "mjonia.teacher.v1", "teacher_id": model_name,
        "teacher_revision": revision, "teacher_weight_license": "Apache-2.0",
        "method": "response-level supervised distillation",
        "student_weights": "independent MJON IA checkpoint v0.7",
        "source": "prompts originales de código propio, respuestas generadas por modelo maestro",
        "warning": "Un modelo pequeño no puede copiar toda la capacidad del maestro; las respuestas requieren evaluación.",
        "training": rows, "holdout": holdout
    }


def save_manifest(path, content):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
