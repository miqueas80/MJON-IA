# Entrenamiento MJON IA v0.2

Este documento describe la versión experimental reproducible. **No mejora por sí
solo la capacidad conversacional:** los datos de demostración son insuficientes.

## En CPU (computadora modesta)

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m mjon_ia.train --data data/demo.txt --out checkpoints/mjon-v02.pt \
  --steps 200 --save-every 50 --eval-every 50 \
  --context 48 --d-model 32 --layers 2 --heads 2 \
  --batch-size 8 --cpu-threads 2 --device cpu
```

Para **reanudar** hasta 400 pasos, mantené dataset, semilla, modo de partición
y configuración del modelo:

```bash
python -m mjon_ia.train --data data/demo.txt --out checkpoints/mjon-v02.pt \
  --resume checkpoints/mjon-v02.pt --steps 400 \
  --save-every 50 --eval-every 50 \
  --context 48 --d-model 32 --layers 2 --heads 2 \
  --batch-size 8 --cpu-threads 2 --device cpu
```

`--steps` significa **pasos totales objetivo**, no pasos adicionales.
El checkpoint conserva pesos, optimizador, generador aleatorio y registro de
métricas. Se guarda atómicamente en `--out`; el JSON de métricas se almacena
junto al checkpoint.

La evaluación utiliza las mismas ventanas de validación en cada registro y
por tanto es comparable a lo largo de una ejecución. No es un benchmark de
razonamiento, lenguaje ni seguridad.

## Particiones

Por defecto `--split-mode dialogues` separa párrafos delimitados por una
línea en blanco, deduplica los bloques exactos y divide el conjunto de bloques
en entrenamiento/validación de manera determinista. A diferencia de la
partición antigua por bytes, un mismo ejemplo literal no queda en ambos
conjuntos. Aun así, existen duplicados semánticos y otros riesgos de
contaminación que deberán auditarse al incorporar datos reales.

`--split-mode bytes` conserva la partición antigua para archivos que no estén
formateados como diálogos. **No es recomendable para evaluar generalización.**

## Compatibilidad

- Los checkpoints v0.1 todavía se pueden utilizar para inferencia.
- Para reanudar se requiere un checkpoint v0.2 completo.
- Las pruebas exactas de reanudación se realizaron en CPU; reanudar con MPS no
  cuenta con garantía de determinismo.
- El exportador de GitHub Pages puede seguir leyendo `model_config`,
  `state_dict` y `step`, que se conservaron.
- Los checkpoints con pesos y estados del optimizador **no se suben al
  repositorio**. No incluyas información privada en datasets.
