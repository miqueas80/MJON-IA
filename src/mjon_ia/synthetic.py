"""Corpus de demostración original en español. No es un corpus conversacional amplio."""
from pathlib import Path
import random

SUBJECTS=['la biblioteca','la escuela','el laboratorio','la plaza','la casa','el museo','el parque','el teatro','la oficina','la estación']
OBJECTS=['libros','mesas','frascos','flores','ventanas','cuadros','árboles','asientos','carpetas','trenes']
COLORS=['rojo','azul','verde','amarillo','negro','blanco','naranja','violeta']
WORDS=['aprender','investigar','estudiar','comprender','imaginar','describir','comparar','observar']


def make_dialogues(seed=42,examples=1500):
    rng=random.Random(seed);blocks=[]
    for n in range(examples):
        a,b=rng.randrange(1,40),rng.randrange(1,40)
        place=rng.randrange(len(SUBJECTS))
        color=rng.choice(COLORS);word=rng.choice(WORDS)
        templates=[
            (f'¿Cuánto es {a} más {b}?',f'{a} más {b} es {a+b}.'),
            (f'Sumá {a} y {b}.',f'El resultado es {a+b}.'),
            (f'¿Cuánto es {a+b} menos {b}?',f'El resultado es {a}.'),
            (f'¿Qué número viene después del {a}?',f'Después del {a} viene el {a+1}.'),
            (f'¿Qué número es mayor, {a} o {b}?',f'El mayor es {max(a,b)}.' if a!=b else 'Ambos números son iguales.'),
            (f'¿Cómo se escribe {a} multiplicado por {b}?',f'El producto {a} por {b} es {a*b}.'),
            (f'Si hay {a} {OBJECTS[place]}, ¿dónde están?',f'Los {a} {OBJECTS[place]} están en {SUBJECTS[place]}.'),
            (f'¿De qué color pueden ser {a} objetos en {SUBJECTS[place]}?',f'Los {a} objetos pueden ser de color {color}.'),
            (f'¿Qué significa {word} en {SUBJECTS[place]}?',f'{word.capitalize()} es una actividad para adquirir o expresar ideas en {SUBJECTS[place]}.'),
            (f'Explicá por qué es útil {word} para {a} estudiantes.',f'{word.capitalize()} ayuda a los {a} estudiantes a desarrollar capacidades y explorar situaciones nuevas.')
        ]
        q,answer=templates[n%len(templates)]
        blocks.append(f'Usuario: {q}\nAsistente: {answer}')
    rng.shuffle(blocks)
    return blocks


def build(output='data/synthetic_es.txt',seed=42,examples=1500):
    if examples<50:raise ValueError('Se necesitan al menos 50 ejemplos')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    records=make_dialogues(seed,examples)
    path.write_text('\n\n'.join(records)+'\n',encoding='utf-8')
    print(f'Corpus sintético: {len(records)} diálogos → {path} ({path.stat().st_size} bytes)')
    return path


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/synthetic_es.txt')
    p.add_argument('--examples',type=int,default=1500)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();build(a.output,a.seed,a.examples)
