# MJON IA v0.8 — Protocolo de destilación auditable

## Qué estamos transfiriendo

La técnica es **destilación por respuestas** (supervised response distillation):
1. El profesor preentrenado responde preguntas de origen controlado.
2. Se registra nombre, revisión exacta y licencia del modelo profesor.
3. Se filtran respuestas incompletas y se aíslan temas de evaluación.
4. Se ajustan los **pesos de MJON IA v0.7** con pares pregunta/respuesta.
5. Se comparan puntajes de v0.7 y v0.8 en pruebas que no se enseñaron.

**No** copiamos capas, matrices, embeddings, vocabulario ni pesos de Qwen al
checkpoint MJON, porque son arquitecturas y representaciones diferentes.
Tampoco declaramos que los nuevos pesos sean equivalentes al profesor.

## Profesor

- Identificador: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Licencia declarada de los pesos: **Apache-2.0**
- Fuente: equipo Qwen; el script guarda el SHA del modelo consumido.
- Dependencia de tiempo de ejecución: **ninguna** para usar MJON v0.8.
- Descarga de pesos: exclusivamente en el ejecutor de GitHub Actions para generar datos.
- Datos personales: no se envían datos de usuarios. Solo prompts de prueba creados en el repositorio.

No etiquetar las respuestas generadas como CC0 automáticamente. La licencia MIT
cubre el código fuente propio; los pesos de Qwen no forman parte del repositorio
ni de los pesos cuantizados MJON. Mantener los avisos y la atribución del maestro
al distribuir material derivado.

## Separación de datos

Hay 48 conceptos originales distribuidos en seis áreas. Se reservan seis
conceptos completos como evaluación; los demás se presentan al profesor como
preguntas de entrenamiento. Los ejemplos no pasan al alumno hasta ser
validados mínimamente en estructura; filtros automáticos **no garantizan
corrección factual**. El corpus y sus metadatos quedan publicados en
\`docs/web/teacher-v08.json\` tras el despliegue.

El alumno conserva su tokenizador propio BPE con 512 tokens, ventana de
256 tokens y aproximadamente 1,29 millones de parámetros. El ajuste utiliza
70% de ejemplos del maestro, 12% de ejercicios de memoria antiguos y
18% de ejercicios escolares antiguos, escogidos durante cada lote.
La comparación escolar usa datos reservados independientes de los del profesor.

## Reproducción

En un entorno CPU con memoria suficiente:

\`\`\`bash
pip install -e .
pip install "transformers>=4.48,<5" "huggingface_hub>=0.27,<2" safetensors
python scripts/generate_teacher_v08.py \
  --out data/teacher_v08.json --train-limit 42 --batch-size 4 --max-new-tokens 40
python -m mjon_ia.distill_v08 --base checkpoints/web/v07.pt \
  --teacher data/teacher_v08.json --out checkpoints/web/v08.pt --steps 480
python scripts/benchmark_v08.py --before checkpoints/web/v07.pt \
  --after checkpoints/web/v08.pt --teacher data/teacher_v08.json
\`\`\`

Hay que generar primero el checkpoint v0.7 mediante el pipeline existente.

## Límites científicos

No hay transferencia instantánea del conocimiento completo de Qwen a MJON:
42 ejemplos son insuficientes para reproducir un modelo de cientos de
millones de parámetros. Si la evaluación independiente no mejora, el
experimento debe reportar ese resultado sin maquillar cifras. La destilación
es un mecanismo de entrenamiento, no una garantía de que el alumno
comprenda el lenguaje natural, razone ni responda con exactitud factual.

La próxima etapa debe contemplar revisión humana de los datos, explicación
por pasos sin filtrar razonamiento oculto de un sistema externo, ampliación
de cobertura y, si se cambia la arquitectura para reutilizar pesos externos,
declarar explícitamente esa variante como **derivada**.
