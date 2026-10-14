# Ficha del modelo — MJON IA v0.1

- **Arquitectura:** Transformer causal autoregresivo implementado en PyTorch.
- **Inicialización:** desde cero, sin pesos de terceros.
- **Tokenizer:** bytes UTF-8 (vocabulario de 256), no BPE.
- **Corpus:** pequeño conjunto sintético `data/demo.txt`, cuya procedencia está descrita en `data/DATA_CARD.md`.
- **Entrenamiento:** `python -m mjon_ia.train`; cada usuario genera el checkpoint localmente.
- **Evaluación:** pérdida de entrenamiento y validación sobre fragmentos de texto. No mide razonamiento ni calidad conversacional.
- **Pesos:** generados localmente, excluidos del repositorio en esta etapa.

**Limitaciones:** resultados incoherentes, corpus insuficiente, falta de evaluaciones extensas y de seguridad robusta. No es apto para decisiones de alto impacto ni como asesor fiable.

## Variante v0.3

- Arquitectura: Transformer causal, pesos desde cero, 505.920 parámetros con configuración d=96, 4 capas, 4 cabezas, vocabulario BPE de 512 unidades.
- Tokenizador BPE entrenado exclusivamente en entrenamiento; reversible para UTF-8.
- Entrenamiento: corpus sintético generado y documentado (licencia CC0), muestras separadas por diálogo, 500 pasos de demostración; checkpoint reanudable disponible durante entrenamiento.
- Inferencia: PyTorch en CLI y JavaScript cuantizado int8 con KV cache en GitHub Pages.
- Limitación crítica: el corpus es limitado y repetitivo. El modelo no tiene razonamiento general ni garantías de veracidad.

## Variante v0.4: Corpus Forge + MJON Bench

- **Parámetros:** 1.271.808, pesos aleatorios al comenzar.
- **Arquitectura:** Transformer causal, 6 capas, d=128, 4 cabezas, contexto 128 tokens.
- **Tokenizador:** BPE UTF-8 propio de hasta 512 tokens, entrenado en un prefijo del conjunto de entrenamiento; sin fuga deliberada de validación.
- **Corpus:** 5000 diálogos originales generados por reglas y semillas documentadas, licencia CC0. Incluye tareas repetitivas de aritmética y conocimientos muy básicos; no es una colección de textos humanos.
- **Entrenamiento:** 1000 pasos, lotes 4 × gradientes acumulados 2, CPU de GitHub Actions.
- **Benchmark:** preguntas sintéticas fuera de plantilla, tasas de respuesta numérica exacta y preferencia correcta frente a un distractor. Reportes de v0.3 y v0.4 en `docs/web/benchmarks.json` tras cada despliegue.
- **Advertencias:** no mide inteligencia general, puede memorizar plantillas o generar errores y no tiene alineamiento instructivo avanzado. Los resultados en pruebas se publican tal como son, sin promesas de mejora.

## Variante v0.5 — Primeras palabras

- **Base:** MJON IA v0.4 (1.271.808 parámetros), sin pesos de terceros.
- **Objetivo SFT:** respuesta breve ante una pregunta que presenta de forma explícita un color, material o estado.
- **Dataset:** 1.360 ejemplos sintéticos CC0, generados localmente.
- **Evaluación:** 102 ejemplos con nombres de objetos que no aparecen en los ejemplos SFT; selección determinista de 48 para el benchmark web.
- **Métrica:** porcentaje de respuestas generadas libremente que coinciden con la etiqueta esperada, sin elegir entre candidatos predefinidos.
- **Limitación:** aprender a extraer una etiqueta no equivale a razonar, conocer el mundo ni dialogar de manera general. Los resultados de v0.4/v0.5 son comparables solo dentro de esta misma prueba.
