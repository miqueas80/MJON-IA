# MJON IA

Proyecto independiente de inteligencia artificial de código abierto en desarrollo. El primer objetivo es construir, entrenar y ejecutar localmente un modelo de lenguaje Transformer pequeño a partir de pesos aleatorios, sin modelos comerciales ni servicios externos.

**Estado:** investigación experimental, no es un asistente conversacional fiable.

## Inicio rápido
```bash
python -m pip install -e .
python -m mjon_ia.train --steps 15 --eval-every 5 --context 32 --d-model 48 --layers 2 --heads 2 --batch-size 4 --device cpu
python -m mjon_ia.generate --model checkpoints/demo.pt --prompt "Usuario: Hola" --tokens 50
python -m unittest discover -s tests -v
```

Requiere Python 3.10+ y PyTorch 2.2+. El entrenamiento crea los pesos en `checkpoints/`, carpeta no versionada; un resultado incoherente es normal con el pequeño dataset de demostración.

## Código y datos
- `src/mjon_ia/`: Transformer causal, entrenamiento, inferencia, datos.
- `data/demo.txt`: corpus sintético de ejemplo (CC0-1.0).
- `tests/`: pruebas reproducibles.
- `docs/`: limitaciones y hoja de ruta.

## Política de desarrollo
Se trabaja en **main**, por partes y con pruebas antes de cada commit. Ante un fallo, usar `git revert <sha>` para conservar el historial.

Código bajo licencia **MIT**. El objetivo de apertura incluye código, pesos entrenados y documentación de datos/licencias cuando estén preparados para publicación. El repositorio es público; los pesos experimentales de la web se generan durante el despliegue. MJON IA no depende de ningún proyecto anterior.


## Probar la IA en GitHub Pages

La interfaz web pública usa **el Transformer real entrenado desde cero**; los
pesos se generan en GitHub Actions y se publican como archivos estáticos.
La inferencia usa JavaScript local, no Python en el servidor.

Guía paso a paso: [docs/GITHUB_PAGES.md](docs/GITHUB_PAGES.md).
Tras activar **Settings > Pages > GitHub Actions**, el sitio esperado es
https://miqueas80.github.io/MJON-IA/ .

El navegador ejecuta un modelo más pequeño (8.176 parámetros) que el
entrenamiento de CLI. Es un experimento de tokens, no un asistente fiable.

## Entrenamiento v0.2

Ahora se puede guardar y reanudar el optimizador y estado aleatorio, medir la validación de forma estable y separar diálogos por bloque para reducir filtración de ejemplos literales entre entrenamiento y validación. **La calidad del chat web no aumenta automáticamente.**

[Instrucciones de entrenamiento v0.2](docs/TRAINING_V02.md).

## Comparar modelos en la web

La web permite alternar entre dos **modelos entrenados por separado**: perfil v0.1 (8.176 parámetros, contexto de 48 bytes) y perfil v0.2 (36.736 parámetros, contexto de 96 bytes). El flujo de GitHub Actions los entrena desde cero, genera pesos cuantizados y construye `docs/web/models.json`. El navegador descarga y ejecuta localmente el modelo seleccionado: no se conecta a una API de terceros para generar texto.

Estas dos variantes usan un dataset minúsculo; **ninguna mantiene una conversación fiable**. Cambiar de versión no sustituye la necesidad de datasets amplios y evaluación. Los pesos no se guardan en `main`: se incluyen en el despliegue de Pages.

## MJON IA v0.3: BPE propio + 505.920 parámetros

Nueva variante entrenada desde pesos aleatorios, tokenizador **byte-level BPE** aprendido únicamente del corpus de entrenamiento, contexto de 96 tokens, gradientes acumulados, checkpoints reanudables y validación separada. Se construye automáticamente desde un corpus sintético propio (CC0). El motor JavaScript tiene **caché de claves/valores** para generar tokens sin repetir innecesariamente la inferencia completa.

Este progreso es **de ingeniería y entrenamiento experimental**, no demuestra razonamiento ni capacidad conversacional general. Los modelos v0.1 y v0.2 siguen disponibles en el selector de GitHub Pages. La construcción se verifica con pruebas de Python y JS; el entrenamiento depende del workflow de GitHub Actions.

Comandos v0.3:

```bash
python -m mjon_ia.synthetic --output data/synthetic_es.txt --examples 1800
python -m mjon_ia.train_v03 --data data/synthetic_es.txt --steps 500 --out checkpoints/mjon-v03.pt --context 96
python -m mjon_ia.generate_v03 --checkpoint checkpoints/mjon-v03.pt --prompt "Usuario: Hola\nAsistente:"
```

Los pesos descargables se publican como parte del sitio web; los checkpoints del optimizador no se suben al repositorio. El objetivo es ampliar después el corpus con materiales de procedencia y licencia verificables.

## MJON IA v0.4 · Corpus Forge + MJON Bench

Esta etapa agrega un cuarto modelo Transformer entrenado desde cero de **1.271.808 parámetros** (d=128, 6 capas, 4 cabezas, contexto de 128 tokens BPE y vocabulario objetivo 512). El corpus sintético CC0 tiene 5.000 diálogos generados de manera determinista a partir de tareas de lenguaje, aritmética, conceptos y estructuras en español. Los archivos derivados se crean en GitHub Actions: el generador y las reglas están en el código fuente.

**Evaluación independiente:** MJON Bench presenta preguntas numéricas redactadas fuera de las plantillas y con números ajenos al rango de entrenamiento. El resultado real se publica en `docs/web/benchmarks.json` y se muestra en el sitio para v0.3/v0.4; las puntuaciones bajas son resultados válidos. No se presenta la validación como garantía de razonamiento ni de calidad conversacional.

El tokenizador se entrena **solo** con un prefijo de la partición de entrenamiento (nunca validación). El entrenamiento conserva checkpoints reanudables y los tres modelos anteriores. El teléfono ejecuta inferencia local en el navegador, mientras la construcción entrena el modelo en GitHub Actions.

```bash
python -m mjon_ia.corpus_forge --examples 5000
python -m mjon_ia.train_v03 --data data/corpus_v04.txt --out checkpoints/v04.pt --steps 1000 --context 128 --vocab-size 512 --tokenizer-chars 80000 --d-model 128 --layers 6 --heads 4 --batch-size 4 --grad-accum 2 --device cpu
python scripts/evaluate_v04.py --checkpoints checkpoints/v04.pt --benchmark data/benchmark_v04.json
```

La investigación sigue siendo experimental. No hay API externa obligatoria ni promesas de equivalencia con un LLM comercial.

## MJON IA v0.5: primeras palabras con supervisión de respuestas

Esta etapa **no agrega otro modelo desde cero**: adapta mediante entrenamiento supervisado (SFT) nuestro Transformer v0.4 de 1.271.808 parámetros, reutilizando sus pesos y el tokenizador propio BPE. La pérdida solo se calcula sobre la respuesta; el prompt no cuenta como objetivo de predicción. Los checkpoints mantienen pesos y estado del optimizador para continuar el ajuste.

**Corpus propio CC0**: 1.360 ejemplos generados determinísticamente sobre color, material y estado. **Evaluación separada**: 102 combinaciones con seis sustantivos ausentes de esos ejemplos, de los cuales MJON First Words examina 48 por generación libre determinista. Publicamos el puntaje real de v0.4 (sin SFT) junto al de v0.5, incluyendo respuestas equivocadas. Se miden únicamente extracciones de atributos explícitos, **no conversaciones abiertas ni razonamiento general**.

La web conserva v0.1-v0.4 y agrega v0.5 con ejemplos apropiados para sus capacidades. GitHub Actions genera los pesos y el informe `docs/web/firstwords.json` durante la publicación.

```bash
python -m mjon_ia.instruction_v05 --out data/instruction_v05
python -m mjon_ia.finetune_v05 --base checkpoints/web/v04.pt --out checkpoints/web/v05.pt
python scripts/benchmark_v05.py --base checkpoints/web/v04.pt --tuned checkpoints/web/v05.pt
```

## MJON IA v0.6 — memoria contextual y correcciones

El mismo Transformer propio de **1.271.808 parámetros** se ajusta desde v0.5 con diálogos sintéticos que presentan un dato y luego preguntan por él, o registran una corrección y preguntan por el valor vigente. El corpus está documentado en `data/MEMORY_V06.md`. Se mezclan ejemplos de repaso de v0.5 para reducir el olvido. Sigue siendo entrenamiento supervisado (SFT), no una base de datos externa ni un sistema general de razonamiento.

**MJON Memory Bench** compara v0.5 y v0.6 con 48 pruebas equilibradas de recuerdo y corrección sobre objetos no incluidos en el ajuste; también mide la retención en el benchmark anterior de 48 extracciones. Las puntuaciones reales se incluyen en `docs/web/memory.json` durante el despliegue de GitHub Pages. Los errores permanecen visibles.

En la web, v0.6 conserva **hasta tres pares de mensajes temporalmente en memoria RAM**, y los incorpora al siguiente prompt. No se guardan entre recargas, no hay reconocimiento fiable de intenciones y no se prometen conversaciones generales. Se preservan v0.1–v0.5 en el mismo selector.

```bash
python -m mjon_ia.memory_v06 --out data/memory_v06
python -m mjon_ia.finetune_v06 --base checkpoints/web/v05.pt --out checkpoints/web/v06.pt
python scripts/benchmark_v06.py --base checkpoints/web/v05.pt --tuned checkpoints/web/v06.pt
```

## MJON IA v0.7 — MJON Escuela (laboratorio escolar)

La v0.7 reutiliza los pesos neuronales originales de v0.6 y **duplica la ventana de contexto de 128 a 256 tokens BPE**, extendiendo las posiciones existentes. No cambia a un modelo de terceros ni introduce servicios de pago.

Se entrenan cuatro familias de ejercicios sintéticos redactados en español: **Matemática** (sumas y restas en secuencia), **Lengua** (comprensión explícita de dos datos), **Ciencias** (magnitudes descritas en fichas) y **Lógica** (orden relativo de tres entidades). Se generan 4.800 problemas de entrenamiento propios CC0, 128 de entidades/números reservados y otros 128 con reformulaciones no presentes en las plantillas de aprendizaje. Se mezclan prácticas de v0.5 y v0.6 para minimizar el olvido.

**Evaluación visible**: el informe `docs/web/school.json` compara v0.6 y v0.7 en 40 problemas reservados, 40 reformulados, más pruebas de retención de memoria y respuestas breves. No se maquilla una puntuación baja: el informe publica ejemplos de aciertos y fallos. La interfaz permite seleccionar los siete modelos históricos y ejecuta la v0.7 localmente.

Este avance no equivale a cursar, aprobar ni comprender la secundaria argentina. El objetivo es un **banco de pruebas multiasignatura y de transferencia** que permita construir capacidades medibles; faltan textos amplios de origen verificable, currículo completo y evaluación humana.

```bash
python -m mjon_ia.school_v07 --out data/school_v07 --per-subject 1200
python -m mjon_ia.train_v07 --base checkpoints/web/v06.pt --out checkpoints/web/v07.pt --steps 1100 --context 256
python scripts/benchmark_v07.py --before checkpoints/web/v06.pt --after checkpoints/web/v07.pt
```

## v0.8 — Alumno de un modelo abierto (destilación real)

Por primera vez, un **profesor abierto externo** genera respuestas sobre ciencia, matemática, lengua e informática para entrenar el **Transformer original de MJON IA**, mediante aprendizaje supervisado solo de la respuesta. El docente es [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), de licencia **Apache 2.0**; sus pesos se descargan **solo** en GitHub Actions durante la construcción y nunca pasan a los pesos del alumno. El origen exacto se guarda con ID del maestro, revisión y ejemplos generados.

El alumno conserva su arquitectura, tokenizer y pesos propios de v0.7. Por diseño **no puede adquirir automáticamente las capacidades de un modelo de 500 millones de parámetros**: se intenta transferir un conjunto limitado de comportamientos y explicaciones en español. El propio profesor puede equivocarse. Nunca afirmamos que MJON sea Qwen ni que tenga su nivel.

El script crea 42 preguntas para enseñar y 6 preguntas de conceptos reservados, descarta respuestas vacías o inadecuadas y mezcla los ejemplos aceptados con repaso escolar y contextual. Se incluyen los ejercicios fuente en `src/mjon_ia/teacher_v08.py`. El alumno resultante se añade como **v0.8** al selector de GitHub Pages y se evalúa frente a v0.7 en un examen escolar independiente y pérdida sobre conceptos reservados. El conjunto del profesor se publica con su procedencia en `docs/web/teacher-v08.json`; la puntuación en `docs/web/distillation.json`.

No se incorpora una dependencia en tiempo de ejecución: la inferencia de v0.8 ocurre **sin internet**, como los anteriores modelos. El acceso a Internet sí es necesario **en el entrenamiento** para descargar el profesor. El proyecto no atribuye automáticamente licencia CC0 a las respuestas generadas por Qwen; el código original sigue bajo MIT y los avisos sobre el modelo externo están documentados.

[Diseño, reproducibilidad y licencias](docs/DISTILLATION_V08.md).

## MJON Universidad · Open-Base (experiencia conversacional derivada)

Además de MJON Original, hay un aula independiente en **[Open-Base](https://miqueas80.github.io/MJON-IA/open-base.html)**. Ejecuta localmente un modelo preentrenado de unos 500 millones de parámetros de Qwen mediante Transformers.js + WebGPU. **No es MJON Original ni representa una transferencia completa de conocimientos**; se identifica claramente la procedencia Apache 2.0 del modelo y se conservan las versiones nativas sin cambios.

La descarga inicial de los pesos cuantizados es optativa y grande (aprox. 483 MB–786 MB más auxiliares); requiere Internet y WebGPU. El navegador puede conservar parte de los recursos en caché, pero no se promete una experiencia 100% offline. Los mensajes no se envían a servicios de inferencia; los cálculos se realizan en el dispositivo después de la carga. El aula incluye evaluaciones humanas exportables, sin calificación automática inventada.

[Protocolo, licencias, requisitos y limitaciones](docs/OPEN_BASE.md).
