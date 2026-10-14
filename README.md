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
