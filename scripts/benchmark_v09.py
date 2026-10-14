"""Evaluación de MJON v0.9 contra v0.8, sin ejecutar Qwen en producción.

La referencia docente viene de un fichero generado durante Actions y las
preguntas reservadas no aparecen como conceptos en el entrenamiento.
"""
from benchmark_v08 import run
import argparse


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="checkpoints/web/v08.pt")
    p.add_argument("--after",default="checkpoints/web/v09.pt")
    p.add_argument("--teacher",default="data/teacher_v09.json")
    p.add_argument("--school-holdout",default="data/school_v07/holdout.json")
    p.add_argument("--school-transfer",default="data/school_v07/transfer.json")
    p.add_argument("--out",default="docs/web/distillation-v09.json")
    p.add_argument("--per-subject",type=int,default=10)
    a=p.parse_args()
    run(a.before,a.after,a.teacher,a.school_holdout,a.school_transfer,a.out,a.per_subject)
