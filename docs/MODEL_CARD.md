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
