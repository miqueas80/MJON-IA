# MJON IA Neural Core — enseñanza de instrucciones sin modelo externo

## Objetivo

El núcleo neuronal propio de MJON ya tiene una arquitectura escalable. Este cambio agrega una **segunda fase de entrenamiento sobre los mismos pesos**, con respuestas supervisadas, para que el modelo aprenda a contestar instrucciones y no únicamente a predecir el siguiente token.

Es una implementación nueva de entrenamiento, **no una segunda IA funcionando en la aplicación**. No se carga Qwen, DeepSeek, Gemini ni un proveedor remoto durante la generación.

## Flujo real de entrenamiento

1. Reunir un corpus de preentrenamiento con textos de procedencia identificada, licencias compatibles y suficiente diversidad; utilizar `source_id`, `document_id` y `license` en JSONL. El pequeño conjunto didáctico incluido solo verifica el proceso.
2. Entrenar el `MJONNeuralCore` desde pesos aleatorios usando `mjon_ia.scalable_train`. Esto produce un checkpoint y un tokenizer SentencePiece entrenado exclusivamente sobre textos de entrenamiento.
3. Reunir pares instrucción–respuesta en JSONL, con `instruction`, `response`, `source_id`, `group_id` y `license`, y ejecutar `mjon_ia.neural_sft`. Se trabaja **sobre el checkpoint propio preentrenado**.
4. El script separa los ejemplos por grupo conceptual/documental, deduplica los exactos, impide que la misma instrucción aparezca en grupos distintos y entrena **solo los tokens de respuesta**; el texto del usuario queda con etiqueta `-100`.
5. Mantiene un conjunto de validación aislado, selecciona el checkpoint de menor pérdida sobre él, permite reanudar con el mismo corpus y parámetros y almacena el estado completo del optimizador y RNG.
6. El examen genera las respuestas **usando exclusivamente los pesos MJON** sobre grupos completamente apartados del entrenamiento, comparando el checkpoint preentrenado con el ajustado.

## Experimento íntegro con material propio

```bash
python -m pip install -e '.[neural]'
python scripts/build_neural_curriculum.py --out data/neural_corpus

# Prueba de integración, pequeña y económica. NO produce una IA potente.
python -m mjon_ia.scalable_train \
  --data data/neural_corpus/knowledge_pretrain.jsonl \
  --tokenizer checkpoints/neural/mjon-bpe.model \
  --out checkpoints/neural/pretrain.pt \
  --vocab-size 512 --context 256 --width 64 --layers 2 \
  --heads 4 --kv-heads 2 --steps 100 --batch-size 4 \
  --grad-accum 2 --warmup 10 --device cpu

python -m mjon_ia.neural_sft \
  --base checkpoints/neural/pretrain.pt \
  --tokenizer checkpoints/neural/mjon-bpe.model \
  --data data/neural_corpus/knowledge_sft.jsonl \
  --out checkpoints/neural/instruct.pt --steps 200 \
  --batch-size 4 --grad-accum 2 --warmup 20 --device cpu

python scripts/eval_neural_sft.py \
  --before checkpoints/neural/pretrain.pt \
  --after checkpoints/neural/instruct.pt \
  --tokenizer checkpoints/neural/mjon-bpe.model \
  --data data/neural_corpus/knowledge_sft.jsonl \
  --out checkpoints/neural/sft-evaluation.json
```

Los números de pasos de este ejemplo son **únicamente una prueba del pipeline**, no una receta de modelo de frontera. Para mejorar en razonamiento, programación y diálogo es necesario un corpus real, extenso y revisado, GPU suficiente y evaluaciones más amplias.

## Entrenamiento en GPU

Los mismos comandos admiten `--device cuda --precision bf16` en GPUs que lo soporten, o `--precision fp16`. La carga de modelos mayores requerirá más VRAM y memoria de datos. **GitHub Actions gratuito no garantiza GPUs ni potencia suficiente** para entrenamiento a escala de un modelo de frontera.

No se conectó ningún servicio pago y no se desplegó ningún modelo sin entrenamiento. La nueva ruta permanece aislada de los pesos web legacy, que continúan operativos.

## Seguridad de datos, evaluación y limitaciones

El JSONL original viene de `src/mjon_ia/knowledge_v11.py`: **540 preguntas de 120 conceptos** (incluye reformulaciones), con atribución CC0 a su redacción propia. Los grupos se separan por concepto antes de entrenar. Ese dataset tiene cobertura muy reducida y no puede hacer que MJON sea equivalente a una IA general de China, EE. UU. u otro país.

El benchmark incluye exactitud literal y las respuestas del modelo. Incluso si baja la pérdida de validación, **solo hablamos de mejora demostrada** cuando la generación libre y pruebas sobre tareas nunca vistas también mejoren sin deteriorar habilidades anteriores.

## Prueba automática

El workflow [`neural-smoke.yml`](../.github/workflows/neural-smoke.yml) ejecuta pruebas de arquitectura y un ciclo completo pequeño de preentrenamiento + SFT + generación y comparación. No instala ni importa modelos externos. La versión de navegador de MJON no se sustituye sin entrenamiento, exportador numéricamente validado y pruebas en el dispositivo.
