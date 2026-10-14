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

Código bajo licencia **MIT**. El objetivo de apertura incluye código, pesos entrenados y documentación de datos/licencias cuando estén preparados para publicación. Por ahora el repositorio es privado, de modo que el código todavía no está publicado. MJON IA no depende de ningún proyecto anterior.
