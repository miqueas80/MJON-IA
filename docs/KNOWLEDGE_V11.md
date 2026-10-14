# MJON Original v1.1 — Conocimiento redactado y aprendizaje continuo

## Por qué

MJON v1.0 probó que preservar habilidades mientras se aprende es posible, pero la red pequeña sigue respondiendo mal a preguntas reformuladas. En lugar de añadir otro modelo o fingir que recibió los pesos de ChatGPT, preparamos una **biblioteca de enseñanza de hechos y límites básicos**, directamente como datos SFT que se convierten en cambios de los pesos entrenados del propio Transformer MJON.

No es un módulo de respuestas precargadas en el navegador: el material solo participa en el entrenamiento y en el informe de evaluación.

## Datos originales

El archivo `src/mjon_ia/knowledge_v11.py` contiene **120 conceptos y definiciones concisas** escritos para este proyecto, de diez áreas: Biología, Química, Física, Matemática, Lengua, Historia y Ciudadanía, Geografía, Informática, Pensamiento Crítico y límites epistémicos. 100 conceptos se enseñan con cuatro preguntas cada uno; los últimos dos de cada materia quedan completamente reservados para las pruebas. La redacción original puede usarse como CC0; eso no afecta las licencias de los pesos y bibliotecas de terceros del pipeline anterior.

Se crean tres archivos reproducibles:
- `train.json`: 360 ejemplos, de 100 conceptos.
- `paraphrase.json`: 90 preguntas con redacción distinta sobre los mismos 100 conceptos. Solo 30 sirven para calibración, otras 60 son el examen final.
- `unknown.json`: 40 preguntas acerca de 20 conceptos que nunca se enseñan mediante gradientes; evalúan lo que el modelo pueda transferir de conocimientos anteriores.

No se usan consultas a modelos externos en esta nueva etapa. Los ejemplos de límites son respuestas sobre datos ausentes; no se debe concluir que el modelo dejará de alucinar en todas las consultas.

## Aprendizaje y selección

- **Modelo alumno**: checkpoint v1.0, exactamente el mismo Transformer MJON, con vocabulario y tamaño de contexto sin cambios.
- **Corpus de entrenamiento**: aproximadamente 42% de nuevos conocimientos, 35% de ejercicios escolares anteriores y 23% de memoria anterior.
- **Protección contra el olvido**: copia congelada de MJON v1.0 como referencia de logits, solo durante el entrenamiento; penalización KL sobre tokens de respuestas antiguas.
- **Selección**: menor pérdida sobre 30 preguntas de desarrollo y al menos el mismo número de aciertos sobre ejercicios antiguos de escuela, transferencia y memoria. Se conserva el checkpoint inicial si no mejora.
- **Examen final**: 70 paráfrasis nuevas, 36 consultas sobre conceptos no enseñados, 40 escolares, 40 reformuladas, 24 de memoria. La selección nunca ve estas muestras finales.

Esta separación ayuda a detectar memorización, pero no garantiza que la red comprenda las explicaciones. Los prompts se generan por patrones y las respuestas dependen de un modelo de ~1,29 millones de parámetros.

## Reproducción

```bash
python -m mjon_ia.knowledge_v11 --out data/knowledge_v11
python -m mjon_ia.learn_v11 --base checkpoints/web/v10.pt --steps 420
python scripts/benchmark_v11.py --before checkpoints/web/v10.pt \
  --after checkpoints/web/v11.pt
```

Todos los archivos derivados se generan con GitHub Actions. El motor del navegador ejecuta solo el checkpoint seleccionado de MJON; ninguna de las respuestas se toma de una tabla durante la inferencia.
