# MJON IA Original v1.0 — Aprender sin olvidar

## Diagnóstico que motivó esta etapa

En una ejecución real, MJON v0.9 obtuvo mejores pérdidas de respuesta a su profesor Qwen que v0.8 (3,89 frente a 5,42). Sin embargo bajó de 11/40 a 6/40 respuestas escolares correctas y de 1/40 a 0/40 en preguntas reformuladas. Este es un caso de **olvido catastrófico**. Más datos docentes, por sí solos, no garantizaron mejora de las capacidades generales.

## Corrección técnica

La v1.0 trabaja **solo sobre el Transformer MJON Original** de 1,29 millones de parámetros:

1. Restaura como punto de partida el checkpoint v0.8 (antes del deterioro de v0.9).
2. Entrena con 30% de ejemplos docentes, 45% de repaso escolar y 25% de memoria conversacional.
3. Ejecuta una copia congelada de MJON v0.8 **solo dentro del entrenamiento** y compara distribuciones de tokens de respuesta antigua mediante KL con temperatura 2. El objetivo del alumno combina cross-entropy de respuestas correctas y la penalización KL para mantener parte del comportamiento anterior.
4. Evalúa cada 70 pasos conceptos docentes reservados, 16 preguntas escolares de calibración, 16 reformuladas y 8 de memoria.
5. Promociona un candidato solo si disminuye al menos 0,01 la pérdida docente y no disminuye los aciertos de calibración respecto a v0.8. Si ninguno cumple, se preserva el checkpoint del paso 0.

**No se incorpora Qwen, Transformers.js, otro motor, API o módulo a la aplicación.** Las dos instancias temporales en entrenamiento son copias del propio MJON y la aplicación recibe únicamente un conjunto final de pesos MJON.

## Distinción entre calibración y examen

La calibración usa índices 10 a 13 de los registros por materia; la evaluación final usa los primeros 10 de cada una, separados. La memoria también se comprueba en el examen completo. En conceptos del profesor, la selección usa la primera mitad del holdout y el examen final la segunda mitad. Ninguna pregunta reservada se utiliza para gradientes.

Si el checkpoint es 0, el resultado **no significa que el alumno aprendió más**, solo que se protegió el modelo anterior al rechazar cambios perjudiciales. Un resultado favorable en desarrollo puede igualmente fallar en examen final: ambos se publican.

## Límites

El alumno sigue siendo pequeño; no puede recibir de golpe el conocimiento completo del maestro Qwen. Los ejercicios, aunque separados por concepto, siguen siendo sintéticos y limitados; no prueban razonamiento general. La distribución anterior de MJON puede ser incorrecta, por lo que imitarla con KL requiere combinarlo con verdaderas etiquetas y evaluación humana. Un guardarraíl de calibración reduce, pero no elimina, el riesgo de pérdida de conocimientos.

## Reproducibilidad

```bash
python -m mjon_ia.distill_v10 \
  --base checkpoints/web/v08.pt \
  --teacher data/teacher_v09.json \
  --replay data/school_v07/train.json \
  --memory data/memory_v06/train.json \
  --out checkpoints/web/v10.pt \
  --steps 350 --teacher-fraction 0.30 --memory-fraction 0.25 --kd-weight 0.6

python scripts/benchmark_v10.py \
  --before checkpoints/web/v08.pt \
  --after checkpoints/web/v10.pt
```

El registro `docs/web/continual-v10.json` publica las respuestas generadas por el alumno, el paso seleccionado y los resultados. La única dependencia externa durante la construcción es el profesor Qwen para producir corpus docente; nunca se usa en inferencia.
