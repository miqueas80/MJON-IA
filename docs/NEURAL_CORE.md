# MJON Neural Core — Escalabilidad real sin sustituir nuestra IA

## Diagnóstico que obliga a cambiar de estrategia

La variante MJON v1.1 obtuvo **0/70** en paráfrasis y **0/40** en conceptos no enseñados. No debemos interpretar la baja de pérdida de entrenamiento como inteligencia general. El proyecto tiene una red educativa extremadamente pequeña, datos reducidos y entrenamiento principalmente en CPU: **no compite con Qwen, DeepSeek ni otros modelos de frontera**.

MJON necesita una etapa distinta: preentrenamiento de lenguaje a escala, un vocabulario suficiente, arquitectura eficiente, datos documentados y evaluaciones de generalización. No alcanza con conectar módulos, grafos o multiplicar preguntas a partir de los mismos conceptos.

## Lo implementado

El archivo `src/mjon_ia/neural_core.py` contiene la siguiente **evolución del propio Transformer MJON**, completamente inicializable con pesos aleatorios:

- RoPE (posiciones rotatorias) sin tabla aprendida de posiciones.
- RMSNorm y SwiGLU para normalización y feedforward.
- Grouped Query Attention (GQA) con menos cabezas KV que cabezas de consulta.
- Atención causal optimizada con `torch.nn.functional.scaled_dot_product_attention` (selecciona la implementación disponible según hardware).
- Caché KV durante generación para evitar recalcular todas las posiciones anteriores; reinicio seguro al alcanzar el contexto.
- Embeddings atados a la proyección de salida, gradientes verificables, configuración reproducible.
- **Ni una sola matriz de pesos de Qwen/DeepSeek ni inferencia externa**.

El entrenador `src/mjon_ia/scalable_train.py` dispone de mixed precision CUDA BF16/FP16, CPU FP32 para pruebas, clipping, acumulación de gradientes, LR coseno con warmup, evaluación consistente, separación por documento, deduplicación, control de licencias, tokenizer SentencePiece BPE entrenado **solo sobre documentos de entrenamiento**, checkpoints completos reanudables y guardado del mejor resultado.

Los modelos de producción MJON v0.1–v1.2 siguen usando `src/mjon_ia/model.py` y `docs/web/runtime.mjs` sin alteraciones. Esta nueva arquitectura **todavía no reemplaza al modelo web**: se publicará únicamente después de entrenar, exportar y validar equivalencia numérica en un runtime compatible. No tiene sentido engañarnos desplegando una red grande con pesos aleatorios.

## Primer experimento y perfiles

```bash
python -m pip install -e '.[neural]'
python -m unittest discover -s tests -p 'test_neural_core.py' -v

# Prueba pequeña con un corpus con licencia y procedencia, en CPU
python -m mjon_ia.scalable_train --data corpus.jsonl \
  --tokenizer checkpoints/neural/demo.model \
  --out checkpoints/neural/demo.pt --vocab-size 512 --context 128 \
  --width 64 --layers 2 --heads 4 --kv-heads 2 \
  --steps 200 --batch-size 4 --grad-accum 2 --device cpu

# Entrenamiento progresivo en GPU CUDA (requiere GPU y corpus real)
python -m mjon_ia.scalable_train --data corpus.jsonl \
  --tokenizer checkpoints/neural/mjon-bpe.model \
  --out checkpoints/neural/mjon-neural.pt --vocab-size 8192 --context 1024 \
  --width 512 --layers 12 --heads 8 --kv-heads 2 \
  --steps 50000 --batch-size 4 --grad-accum 16 \
  --warmup 1000 --lr 0.0003 --device cuda --precision bf16

# Si una ejecución se interrumpe, reanudar con la MISMA meta de pasos y parámetros:
python -m mjon_ia.scalable_train ... --resume checkpoints/neural/mjon-neural.last.pt
```

Esos comandos son configuraciones de ingeniería, **no afirmaciones de entrenamiento completado ni garantía de calidad**. Una GPU CUDA compatible y una cantidad significativa de datos legales y variados son requisitos reales; GitHub Actions estándar no ofrece entrenamiento masivo gratuito de frontera. El script actual procesa el corpus en memoria RAM y sirve para la primera etapa de escalamiento; entrenamiento de corpus a escala web requerirá streaming/memmap y ejecución distribuida.

## Formato obligatorio del corpus

Cada línea de `corpus.jsonl` debe ser un registro JSON autónomo:

```json
{"text":"Un documento original en español con información verificable.", "source_id":"mjon-author", "document_id":"doc-0001", "license":"CC0-1.0"}
```

El identificador documental mantiene fragmentos del mismo documento dentro del mismo split, mientras SHA-256 deduplica textos iguales. Solo se admiten licencias de la lista explícita. Esto no verifica jurídicamente que quien entrega un archivo tenga los derechos; sigue siendo responsabilidad de la curación del corpus. No subir materiales privados sin autorización.

## Cuándo hablar de una IA potente

Se evaluará, con pruebas nunca vistas y revisadas, si el nuevo modelo:
- Responde a reformulaciones y a conceptos no repetidos.
- Resuelve tareas de matemática, ciencia y programación.
- Conserva contexto y reconoce sus limitaciones.
- Evita inventar datos y muestra mejoras consistentes respecto de MJON v1.1.
- Puede ejecutarse con latencia y memoria aceptables en los dispositivos elegidos.

**Paridad con los mejores modelos chinos:** no demostrada ni técnicamente alcanzable hoy con 1,3M de parámetros y CPU gratuito. Este trabajo abre una ruta auténtica para aumentar escala y calidad sin dejar de ser MJON; el siguiente cuello de botella es conseguir datos autorizados y GPUs suficientes.
