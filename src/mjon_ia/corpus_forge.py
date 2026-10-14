"""Corpus Forge v0.4: diálogos sintéticos originales y pruebas fuera de plantilla.

CC0-1.0. No incorpora textos web, datos de terceros ni contenidos privados.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

PLACES = ("biblioteca", "escuela", "laboratorio", "museo", "estación", "teatro", "parque", "observatorio")
THINGS = ("libros", "cuadernos", "mesas", "cajas", "lápices", "mapas", "frascos", "plantas")
COLORS = ("verde", "azul", "rojo", "amarillo", "violeta", "naranja", "blanco", "negro")
VERBS = ("comparar", "describir", "examinar", "organizar", "calcular", "investigar", "registrar")
DEFS = {
    "célula": "La célula es una unidad estructural y funcional de los seres vivos.",
    "átomo": "Un átomo es una unidad de materia formada por un núcleo y electrones.",
    "ecosistema": "Un ecosistema reúne organismos y el ambiente con el que interactúan.",
    "energía": "La energía es una magnitud asociada a la capacidad de producir cambios.",
    "molécula": "Una molécula es un conjunto de átomos enlazados.",
    "algoritmo": "Un algoritmo es una secuencia ordenada de instrucciones.",
    "variable": "Una variable es un nombre asociado a un valor que puede cambiar.",
    "fracción": "Una fracción representa una parte o cociente de cantidades.",
}

def record(question, answer):
    if not question.strip() or not answer.strip():
        raise ValueError("Registro vacío")
    return f"Usuario: {question}\nAsistente: {answer}"

def generate(seed=2026, examples=5000):
    """Genera ejemplos reproducibles con diversidad de operaciones y formulaciones."""
    if examples < 120 or examples > 200000:
        raise ValueError("examples debe estar entre 120 y 200000")
    rng = random.Random(seed)
    results, seen = [], set()
    tries = 0
    while len(results) < examples:
        tries += 1
        if tries > examples * 50:
            raise ValueError("No se pudieron generar registros únicos")
        a, b, c = (rng.randint(2, 75) for _ in range(3))
        place, item = rng.choice(PLACES), rng.choice(THINGS)
        color, verb = rng.choice(COLORS), rng.choice(VERBS)
        term = rng.choice(tuple(DEFS))
        case = tries % 16
        examples_by_type = [
            (f"Sumá {a} y {b}.", f"El resultado de la suma es {a+b}."),
            (f"¿Cuánto es {a} más {b}?", f"{a} más {b} da {a+b}."),
            (f"Restá {b} a {a+b}.", f"El resultado es {a}."),
            (f"¿Cuánto es {a} por {b}?", f"{a} por {b} es {a*b}."),
            (f"Completá la secuencia {a}, {a+1}, {a+2}.", f"El siguiente número es {a+3}."),
            (f"Entre {a} y {b}, ¿cuál es mayor?", f"El mayor es {max(a,b)}." if a != b else "Los números son iguales."),
            (f"En {place} hay {a} {item} y llegan {b} más. ¿Cuántos hay?", f"Ahora hay {a+b} {item}."),
            (f"Una caja tiene {a} objetos. Se retiran {b if b<=a else a} objetos. ¿Cuántos quedan?", f"Quedan {a-min(a,b)} objetos."),
            (f"¿Qué es {term}?", DEFS[term]),
            (f"Definí el término {term}.", DEFS[term]),
            (f"Explicá en pocas palabras qué significa {verb}.", f"{verb.capitalize()} permite trabajar con información de una manera organizada."),
            (f"¿Qué color tiene el objeto marcado como {color}?", f"El color indicado es {color}."),
            (f"En {place} se usan {a} {item}. ¿Qué objetos se usan?", f"Se usan {item}."),
            (f"Si tenés {a} y agregás {b} y luego {c}, ¿cuál es el total?", f"El total es {a+b+c}."),
            (f"¿Es {a} mayor que {b}?", "Sí." if a>b else "No."),
            (f"¿Cuánto es el doble de {a}?", f"El doble de {a} es {2*a}."),
        ]
        q, ans = examples_by_type[case]
        entry = record(q, ans)
        if entry not in seen:
            seen.add(entry)
            results.append(entry)
    rng.shuffle(results)
    return results

def benchmark(seed=7102, count=36):
    """Preguntas de formulación distinta y números fuera de la franja de entrenamiento."""
    rng = random.Random(seed)
    tests, used = [], set()
    cases = ("addition", "subtraction", "double", "greater", "sequence", "definition")
    while len(tests) < count:
        kind = cases[len(tests)%len(cases)]
        a,b=rng.randint(80,180),rng.randint(10,70)
        if kind=="addition": q=f"¿Qué valor resulta de {a} + {b}?";answer=str(a+b)
        elif kind=="subtraction": q=f"Resolvé {a+b} - {b}.";answer=str(a)
        elif kind=="double": q=f"¿Cuál es dos veces {a}?";answer=str(a*2)
        elif kind=="greater": q=f"¿Cuál de estos números es más grande: {a}, {b}?";answer=str(max(a,b))
        elif kind=="sequence": q=f"¿Qué sigue tras {a}, {a+1} y {a+2}?";answer=str(a+3)
        else:
            term=rng.choice(tuple(DEFS))
            q=f"Explicá brevemente la idea de {term}.";answer=DEFS[term]
        if q not in used:
            used.add(q);tests.append({"task":kind,"question":q,"expected":answer})
    return tests

def build(corpus="data/corpus_v04.txt", probes="data/benchmark_v04.json", examples=5000, seed=2026):
    texts=generate(seed=seed,examples=examples)
    output=Path(corpus);output.parent.mkdir(parents=True,exist_ok=True)
    content="\n\n".join(texts)+"\n"
    output.write_text(content,encoding="utf-8")
    sample=benchmark()
    if any("Usuario: "+row["question"]+"\n" in content for row in sample):
        raise ValueError("Los probes no deben figurar literalmente en el entrenamiento")
    dest=Path(probes);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({"version":1,"seed":7102,"probes":sample},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Corpus v0.4: {len(texts)} diálogos, {len(content.encode('utf-8'))} bytes, sha256={hashlib.sha256(content.encode()).hexdigest()}")
    print(f"Benchmark: {len(sample)} preguntas originales fuera de plantilla")
    return output,dest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",default="data/corpus_v04.txt")
    p.add_argument("--probes",default="data/benchmark_v04.json")
    p.add_argument("--examples",type=int,default=5000)
    p.add_argument("--seed",type=int,default=2026)
    a=p.parse_args()
    build(a.corpus,a.probes,a.examples,a.seed)
