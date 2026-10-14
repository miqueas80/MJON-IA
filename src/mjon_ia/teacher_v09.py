"""Preguntas originales para destilación ampliada; NO hay un motor externo en MJON IA.

Los conceptos de evaluación están separados de entrenamiento.
La IA maestra Qwen responde durante GitHub Actions; los pesos de MJON son propios.
"""
import random

from .teacher_v08 import TOPICS, verify_separation


def variants(question):
    core=question.strip().rstrip("?.")
    return (
        question,
        f"Contestá esta pregunta con precisión: {question}",
        f"En pocas palabras, {question[0].lower()+question[1:]}",
        f"Me gustaría aprender esto: {question}",
        f"Respondé con una explicación sencilla: {question}",
        f"Para una clase de secundaria: {question}",
        f"En una oración, explicá: {core}.",
        f"Explicá el siguiente concepto sin rodeos: {core}.",
    )


def build_prompts(seed=909, variants_per_concept=8, holdout_variants=3):
    if not 1<=variants_per_concept<=8 or not 1<=holdout_variants<=8:
        raise ValueError("Número de reformulaciones fuera de rango")
    training, holdout=[],[]
    for subject, concepts in TOPICS.items():
        for concept_index, (train_question, heldout_question) in enumerate(concepts):
            concept=f"{subject}:{concept_index}"
            if concept_index==7:
                prompts=variants(heldout_question)[:holdout_variants]
                destination=holdout
            else:
                prompts=variants(train_question)[:variants_per_concept]
                destination=training
            for phrasing,question in enumerate(prompts):
                destination.append({"question":question,"subject":subject,
                                    "concept":concept,"phrasing":phrasing})
    random.Random(seed).shuffle(training)
    random.Random(seed+1).shuffle(holdout)
    verify_separation(training,holdout)
    if len(training)!=42*variants_per_concept or len(holdout)!=6*holdout_variants:
        raise AssertionError("Número de ejemplos inesperado")
    return training,holdout
