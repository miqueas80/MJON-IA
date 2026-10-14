# MJON IA v0.9 — Entrenamiento ampliado con un profesor abierto

## Objetivo autorizado

Hacer que **MJON IA Original aprenda** con un modelo de código abierto, sin instalar dicho modelo como módulo en la interfaz, sin crear otra IA o una universidad, y sin introducir una segunda aplicación.

El maestro [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) tiene licencia Apache 2.0 para sus pesos. Se usa **exclusivamente durante la construcción en GitHub Actions**, donde genera un corpus de enseñanza. El alumno es nuestro Transformer de ~1,29 millones de parámetros; sus pesos se actualizan con pérdida supervisada únicamente en el texto de respuesta.

Al visitar MJON IA **no se ejecuta ni descarga Qwen**. El modelo publicado son solamente los pesos int8 de nuestro Transformer, ejecutados por el motor JavaScript original.

## Datos, independencia y procedencia

- 42 conceptos docentes en seis áreas; cada uno usa ocho redacciones originales. Hasta 336 ejemplos de entrenamiento antes de filtrar por longitud y calidad estructural.
- Seis conceptos completos reservados, con tres redacciones cada uno (hasta 18 preguntas de evaluación). Nunca se usan sus respuestas para gradientes.
- El docente genera las respuestas con su tokenizer y la versión exacta del repositorio publicada por Hugging Face. El corpus anota nombre, revisión SHA, licencia y material textual.
- Ninguna muestra recibe automáticamente licencia CC0 por ser generada por el docente. Las preguntas de origen MJON son propias, pero las respuestas requieren revisión humana antes de enseñar contenidos curriculares.
- Las muestras demasiado largas para la ventana del alumno de 256 tokens se rechazan, sin truncar silenciosamente la respuesta.

## Entrenamiento y protección contra regresión

Se parte del checkpoint genuino **MJON v0.8** ya entrenado. Cada lote incluye ejemplos docentes (65%), memoria (12%) y ejercicios escolares previos (23%). Cada 100 pasos se mide:
1. Pérdida sobre conceptos no entrenados.
2. Pérdida de repaso escolar.
3. Pérdida de repaso de memoria.

Se selecciona y publica **el checkpoint con mejor puntuación compuesta**. Se guarda el último checkpoint separadamente para reanudar entrenamiento. El checkpoint v0.9 solo contiene parámetros del mismo Transformer MJON, metadatos y estadísticas. Ni siquiera el formato permite incrustar matrices Qwen incompatibles.

La comparación final confronta v0.8 y v0.9 en las preguntas reservadas y en dos exámenes escolares independientes. Nunca se atribuye a MJON conocimiento que no haya demostrado.

## Límites reales

- Este experimento no transfiere la totalidad de las capacidades de Qwen. Su tamaño es órdenes de magnitud superior.
- 336 preguntas con reformulaciones son una mejora de cobertura, pero siguen representando solo 42 conceptos y pueden inducir memorización de patrones.
- Un resultado favorable en pérdida de lenguaje no demuestra comprensión general. También deben mejorar respuestas libres, nuevos contextos y evaluación humana.
- El código abierto, licencia y atribución del profesor no significan que los pesos Qwen pasen a formar parte del modelo MJON Original.

## Reproducibilidad

```bash
python scripts/generate_teacher_v09.py --student-base checkpoints/web/v08.pt \
  --out data/teacher_v09.json --variants-per-concept 8 --holdout-variants 3
python -m mjon_ia.distill_v09 --base checkpoints/web/v08.pt \
  --teacher data/teacher_v09.json --out checkpoints/web/v09.pt --steps 700
python scripts/benchmark_v09.py --before checkpoints/web/v08.pt \
  --after checkpoints/web/v09.pt --teacher data/teacher_v09.json
```

Las bibliotecas `transformers` y `huggingface_hub` solo se instalan dentro del job que genera los datos. No hay nuevos paquetes en el navegador, ni módulos de inferencia externa, ni backend obligatorio.
