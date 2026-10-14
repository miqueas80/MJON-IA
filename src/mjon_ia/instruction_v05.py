"""MJON IA v0.5: diálogos sintéticos originales (CC0). Validación por objetos nuevos."""
import json
from pathlib import Path
import random

TRAIN_ITEMS=("frasco","lápiz","libro","envase","vaso","termómetro","cuaderno","cajón","recipiente","bolígrafo","cartel","objeto","panel","estante","tubo","contenedor","equipo","papel","tarro","instrumento")
TEST_ITEMS=("astrolabio","periscopio","alicate","telescopio","anemómetro","compás")
ATTRIBUTES={
    "color":("rojo","azul","verde","amarillo","negro","blanco","violeta","naranja"),
    "material":("madera","metal","plástico","vidrio","cartón","cerámica"),
    "estado":("sólido","líquido","gaseoso"),
}

def question(kind,item,value,variant):
    if kind=="color":
        forms=("Ficha del {item}: color {value}. ¿Qué color indica la ficha?",
               "El {item} figura con color {value}. Decime el color.",
               "Leé esto: {item}; color: {value}. ¿Cuál es el color?",
               "Registro del {item}. Color = {value}. Respondé el color.")
    elif kind=="material":
        forms=("Ficha del {item}: material {value}. ¿De qué material es?",
               "El {item} es de {value}. Decime el material.",
               "Leé esto: {item}; material: {value}. ¿Cuál es el material?",
               "Registro del {item}. Material = {value}. Respondé el material.")
    elif kind=="estado":
        forms=("Ficha del {item}: estado {value}. ¿Qué estado indica?",
               "El {item} está en estado {value}. Decime el estado.",
               "Leé esto: {item}; estado: {value}. ¿Cuál es el estado?",
               "Registro del {item}. Estado = {value}. Respondé el estado.")
    else:raise ValueError("Categoría desconocida")
    return forms[variant].format(item=item,value=value)

def build_dataset(seed=504,limit=6000):
    if limit<50:raise ValueError("Se necesitan al menos 50 ejemplos")
    records=[]
    for kind,values in ATTRIBUTES.items():
        for item in TRAIN_ITEMS:
            for value in values:
                for variant in range(4):
                    records.append({"question":question(kind,item,value,variant),"answer":value,"task":kind})
    rng=random.Random(seed);rng.shuffle(records)
    training=records[:min(limit,len(records))]
    holdout=[]
    for kind,values in ATTRIBUTES.items():
        for item in TEST_ITEMS:
            for index,value in enumerate(values):
                holdout.append({"question":question(kind,item,value,index%4),"answer":value,"task":kind})
    rng.shuffle(holdout)
    assert set(x["question"] for x in training).isdisjoint(x["question"] for x in holdout)
    assert all(not any(item in x["question"] for item in TEST_ITEMS) for x in training)
    return training,holdout

def write_dataset(directory,seed=504,limit=6000):
    train,holdout=build_dataset(seed,limit)
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    for filename,values in (("train.json",train),("holdout.json",holdout)):
        (out/filename).write_text(json.dumps(values,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return train,holdout

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/instruction_v05")
    p.add_argument("--seed",type=int,default=504)
    p.add_argument("--limit",type=int,default=6000)
    args=p.parse_args()
    train,held=write_dataset(args.out,args.seed,args.limit)
    print(f"Primeras palabras: {len(train)} ejemplos de entrenamiento, {len(held)} casos reservados")
