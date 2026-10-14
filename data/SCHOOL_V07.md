# MJON Escuela v0.7 — Datos de entrenamiento y evaluación

## Procedencia y licencia

**Datos originales sintéticos**, generados mediante reglas reproducibles por
`src/mjon_ia/school_v07.py`. CC0-1.0 para el contenido de ejemplo, sin copiar
libros, evaluaciones comerciales ni información personal. Los conjuntos
generados se excluyen de Git y se regeneran en GitHub Actions.

## Asignaturas y alcance

| Área | Ejercicios | Qué NO demuestran |
| --- | --- | --- |
| Matemática | sumar y restar cantidades dentro de enunciados | álgebra, demostraciones ni razonamiento simbólico general |
| Lengua | recuperar un color desde dos descripciones | comprensión lectora de textos largos |
| Ciencias | identificar magnitudes explícitas en una ficha | conocimientos de física, química o biología no presentes en la ficha |
| Lógica | ordenar tres nombres según precedencia | deducción formal abierta |

El corpus usa 4.800 registros de entrenamiento (1.200 por asignatura).
Hay 128 ejemplos reservados sobre nuevos números o entidades y otros 128
con formulaciones nuevas. El examen web toma 10 casos por asignatura
de cada uno de esos dos conjuntos (80 consultas por modelo).

El tokenizador reutiliza el BPE propio de v0.6; no se entrena con las respuestas
del examen. El ajuste supervisa únicamente los tokens de respuesta del
asistente y mezcla tareas antiguas para mitigar el olvido.

## Interpretación correcta

La existencia de cuatro áreas de ejercicio **no hace que MJON IA tenga
conocimientos de secundaria**. Es una etapa experimental para aprender
métodos y medir transferencia, incluso cuando el resultado es 0%. Las
plantillas de base siguen siendo estrechas y las pruebas sintéticas no
sustituyen evaluación humana sobre material curricular real.

Reproducción:

```bash
python -m mjon_ia.school_v07 --out data/school_v07 --per-subject 1200 --heldout 32
python -m mjon_ia.train_v07 --base checkpoints/web/v06.pt --context 256
python scripts/benchmark_v07.py --before checkpoints/web/v06.pt --after checkpoints/web/v07.pt
```
