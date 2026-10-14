# MJON Memory v0.6 — ficha de datos sintéticos

Corpus original del proyecto, generado mediante reglas y semillas reproducibles. **CC0-1.0** en la medida permitida por ley. Ninguna conversación se extrae de usuarios, escuelas o páginas de Internet.

## Objetivo y construcción

Entrenar un pequeño Transformer autoregresivo propio para que pueda **recordar atributos mencionados en mensajes anteriores** y **dar prioridad a una corrección posterior**. Hay dos clases de preguntas: recuerdo de un dato inicial y actualización del dato vigente. También se enseñan confirmaciones cortas ("Anotado" y "Actualizado") para intercambios de prueba.

Se generan los ejemplos así:

```bash
python -m mjon_ia.memory_v06 --out data/memory_v06 --seed 606 --limit 2500
```

Se usan tres tipos de atributo (color, material, estado), valores explícitos y nombres de objetos. El corpus se separa por **nombre del objeto**, no mediante división aleatoria de mensajes individuales: nombres de `TEST_ITEMS` no aparecen como objetos de entrenamiento. Las preguntas de evaluación contienen otros objetos y no están duplicadas literalmente en el corpus SFT.

Se usan además registros de aprendizaje de v0.5 como repaso (20% de lotes), para medir y amortiguar el olvido de capacidades anteriores.

## Métricas y limitaciones

`scripts/benchmark_v06.py` evalúa 48 preguntas equilibradas (24 de recuerdo y 24 de corrección; colores, materiales y estados). Compara v0.5 y v0.6, publicando respuestas generadas y cifras exactas. También registra si la v0.6 conserva el desempeño de v0.5 en otras 48 preguntas.

Este benchmark **no mide memoria persistente**, comprensión de conversaciones arbitrarias ni razonamiento general. Las formulaciones, los valores y las categorías permanecen limitados a plantillas conocidas. Los números solo valen para este experimento controlado.

Los datos generados se ignoran por Git: se recrean en Actions y no contienen información personal. Para pasar a textos reales se necesitará registrar procedencia, permisos, licencias y privacidad.
