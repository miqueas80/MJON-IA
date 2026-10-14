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

## Variante v0.6 — Memoria contextual de corto alcance

- **Base:** pesos de MJON IA v0.5 (1.271.808 parámetros), sin LLM ajeno.
- **Entrenamiento:** SFT sobre diálogos sintéticos con turnos previos, cambios de atributo y acuses de recibo; repaso de ejemplos v0.5.
- **Evaluación:** 48 casos equilibrados de recuerdo/corrección con seis nombres reservados en entrenamiento; comparación directa v0.5/v0.6, más retención de 48 preguntas de v0.5.
- **Inferencia web:** historial temporal de hasta tres intercambios enviado únicamente al propio modelo ejecutado en JavaScript en el navegador. No persiste al recargar.
- **Advertencias:** las plantillas del corpus son conocidas y la memoria está limitada por una ventana de 128 tokens BPE. La evaluación no representa diálogo libre, memoria duradera ni razonamiento general.

## v0.7 — MJON Escuela

- **Origen:** mismo Transformer propio de v0.6 con pesos reentrenados, sin checkpoints externos. Ventana extendida a 256 tokens BPE, aprox. 1,29M parámetros; las nuevas posiciones parten de una copia cíclica de las anteriores.
- **Objetivos sintéticos:** 4 materias con problemas elementales en español y pérdida supervisada solo en la respuesta.
- **Conjuntos separados:** números nuevos para Matemática, sustantivos reservados para Lengua/Ciencias/Lógica; más un segundo banco con estructuras alternativas.
- **Evaluación:** generación libre greedy, exactitud por materia y modalidad, regresión de recuerdos y extracción simple frente a v0.6.
- **Limitaciones:** conjunto sintético reducido; las preguntas son ejercicios introductorios, no equivalen a secundaria real, conocimiento enciclopédico ni razonamiento formal. Un resultado bueno en plantillas conocidas no implica generalización a lenguaje natural abierto.

## v0.8 — Destilación desde Qwen2.5-0.5B-Instruct (profesor externo)

- **Alumno:** Transformer MJON v0.7, mismo tokenizer BPE, mismos parámetros y arquitectura, finetune sobre ejemplos del maestro.
- **Maestro:** modelo de pesos abiertos `Qwen/Qwen2.5-0.5B-Instruct` con licencia Apache-2.0. No es parte del modelo final ni se incrustan sus pesos.
- **Método:** destilación a nivel de respuesta por supervisión SFT; no usa KL token a token, porque los tokenizadores son diferentes.
- **Corpus:** preguntas originales controladas, respuestas generadas en CI con la revisión exacta del maestro y publicadas con procedencia. Las respuestas deben revisarse antes de reutilizarse como contenidos educativos avalados.
- **Límites:** con 1,3M de parámetros el alumno no reproduce la capacidad del maestro de ~500M; mayor similitud con el maestro no demuestra razonamiento ni veracidad.
- **Benchmark:** pérdida de respuesta para conceptos reservados + exactitud escolar con problemas y formulaciones no presentes en la muestra del maestro. Registra resultados antes/después y no garantiza mejora.

## v0.9 — Destilación ampliada en el modelo original MJON

- **Arquitectura del alumno:** exactamente la de MJON v0.8 (~1,29 millones de parámetros, contexto 256, BPE propio); se modifican sus pesos entrenables, no se añade arquitectura ni backend.
- **Modelo maestro exclusivamente de entrenamiento:** Qwen2.5-0.5B-Instruct, Apache 2.0, con revisión y procedencia registradas. No se publica ni se carga al ejecutar MJON IA.
- **Datos docentes:** hasta 340 preguntas con ocho redacciones de 42 conceptos. Evaluación: 18 preguntas acerca de seis conceptos separados.
- **Aprendizaje:** SFT sobre respuestas breves del maestro, alternado con repaso de tareas previas. Guarda el mejor checkpoint y el último por separado.
- **Evaluación:** pérdida de respuesta en conceptos reservados, regresión escolar y transferencia sobre redacciones no entrenadas. El profesor puede equivocarse y las etiquetas no son un currículo humano revisado.
- **Capacidad y límites:** no se transfiere de golpe la capacidad de un modelo de 500 millones de parámetros a otro de 1,3 millones; la mejora depende del entrenamiento y de exámenes independientes.

## v1.0 — protección contra el olvido

- Arquitectura: sin cambios frente a v0.8, aprox. 1,29 millones de parámetros, 256 tokens y BPE propio.
- Docencia: Qwen abierto únicamente durante la generación de material de entrenamiento.
- Técnica: repaso de competencias antiguas más destilación funcional de logits entre el propio MJON congelado v0.8 y el alumno actualizado.
- Calibración: cuatro preguntas escolares por materia después de las primeras diez reservadas al examen final; ocho de memoria.
- Publicación: un checkpoint solo se acepta si mejora la pérdida docente y no reduce aciertos en las categorías de calibración. En caso contrario se retienen los pesos de base.
- Limitaciones: preservar la calibración no garantiza preservar todas las habilidades. El examen final se registra sin ocultar retrocesos ni exagerar equivalencia con el maestro.

## v1.1 — currículo original de conocimiento fundamental

- Arquitectura sin cambios: Transformer MJON, aproximadamente 1,29 millones de parámetros y 256 tokens de contexto.
- Datos: 120 conceptos con respuestas originales en nueve materias, 90 enseñados y 18 reservados, más ejemplos de repaso anteriores.
- Entrenamiento: cross-entropy sobre respuestas y regularización KD con una copia congelada del propio modelo. No requiere un profesor externo.
- Separación: 30 paráfrasis de calibración, 70 paráfrasis de examen y 36 consultas de conceptos nuevos; examen escolar y memoria independientes.
- Publicación: solo se acepta checkpoint que reduzca pérdida en calibración de conocimiento y no retroceda en aciertos previos de calibración.
- Advertencia: una definición recordada no prueba comprensión ni razonamiento general; ningún contenido histórico, científico o escolar está avalado automáticamente para evaluación académica.

## v1.2 — Supervisión de relaciones Graphify

- **Estudiante:** mismo Transformer MJON con tokenizer y arquitectura propios; checkpoint v1.1 como base.
- **Origen estructural:** `Graphify-Labs/graphify` / `graphifyy==0.9.90` (Apache-2.0), usado en GitHub Actions únicamente para análisis AST local.
- **Entrenamiento:** pares pregunta-respuesta derivados de aristas `EXTRACTED` del grafo de código, positivos y negativos de ausencia de registro. Pérdida supervisada sobre tokens de respuesta; mezcla de currículo, memoria y escuela.
- **Evaluación independiente:** aristas de prueba no utilizadas para gradientes ni selección, repaso escolar y memoria; progreso rechazado ante regresión de calibración.
- **Limitación:** no hay nueva red neuronal gráfica, «conocimiento universal» ni acceso al grafo en la inferencia. No se pueden convertir aristas de código en capacidades generales por sí solas.

## Modo local verificable: ficha de capacidades separada de los pesos

La entrega funcional añade una capa de conocimientos explícitos en `docs/web/knowledge-core.json` (120 conceptos originales CC0), un intérprete aritmético restringido y memoria temporal. **Estas respuestas no se generan mediante el Transformer neuronal MJON** y no deben contarse como aciertos de evaluación neuronal. Están etiquetadas con fuente y mecanismo.

La capacidad neuronal histórica sigue siendo experimental y cuenta con sus benchmarks propios, separados de la prueba funcional. La nueva experiencia local no justifica afirmar equivalencia con DeepSeek o Qwen. Comprobar tanto los tests automáticos como el uso móvil real, incluyendo caché offline, antes de venderlo como producto.
