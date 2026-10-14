# Herencia neuronal de MJON IA

El usuario dispone de **una sola aplicación MJON IA** y un único selector que incluye dos tipos de cerebro:

- **MJON Original v0.1–v0.8:** Transformer propio entrenado desde pesos aleatorios, arquitectura pequeña y aprendizaje por destilación SFT con ejemplos del maestro abierto en v0.8.
- **MJON IA · cerebro heredado:** usa pesos *preentrenados* de [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (~494M parámetros) a través de la conversión [ONNX Community](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct), licencia Apache 2.0. Esta opción tiene capacidad conversacional desde su preentrenamiento y **es una variante derivada de Qwen**, no un logro del entrenamiento del Transformer MJON Original.

No se puede introducir directamente 494M de parámetros en las matrices de una red de 1,3M: las arquitecturas, tamaños y tokenizadores no coinciden. Para transmitir competencias hay dos métodos técnicamente diferentes: **destilación** de respuestas del maestro hacia el Transformer original, que ya existe pero no consigue replicar su capacidad, y **herencia de pesos preentrenados**, utilizada en el modo heredado, que ofrece potencia inmediata con atribución.

## Cómo funciona

1. Abrí la página principal de MJON IA y seleccioná `MJON IA · Cerebro heredado (Qwen 0.5B)`.
2. Elegí `q4f16` (~483 MB) o `q4` (~786 MB). Ambos pueden requerir archivos auxiliares.
3. Pulsá `Heredar inteligencia`. Solo entonces se solicita descargar el modelo y Transformers.js.
4. Si WebGPU y la memoria disponibles son compatibles, MJON IA podrá usar ese cerebro en la misma interfaz de chat.
5. Se puede volver a cualquier versión original desde el selector, sin borrar sus capacidades.

**Privacidad:** una vez descargado, la generación utiliza un Web Worker/WebGPU local. El navegador accede a un CDN y a Hugging Face durante la descarga; no hay solicitudes de inferencia a APIs. Los chats no se guardan deliberadamente en un servidor. La caché del navegador puede conservar archivos, pero no se garantiza disponibilidad sin Internet hasta haber probado una sesión totalmente desconectada.

**Recursos:** descarga inicial grande. En un teléfono con Snapdragon 8 Gen 1, WebGPU/memoria disponible pueden variar según navegador; la compatibilidad no se garantiza sin probarla. Si no hay WebGPU, las versiones MJON Original siguen funcionando.

## Entrenamiento posterior

La **destilación SFT** de v0.8 se conserva para que MJON Original siga aprendiendo con ejemplos de un profesor abierto. Para ajustar directamente el cerebro heredado habrá que entrenar un adaptador (p. ej. LoRA) sobre Qwen2.5-0.5B-Instruct en un entorno adecuado, validar su calidad y exportar/adaptar ese resultado a ONNX de forma reproducible. **Esta versión usa los pesos publicados sin ajuste LoRA propio todavía**: nunca presentar la herencia como fine-tuning completado.

Las pruebas automáticas comprueban que el modo existe y no afecta al motor MJON Original. No prueban que una descarga de cientos de MB ni WebGPU vayan a funcionar en todos los teléfonos. La publicación se considera lista solo cuando GitHub Actions termina correctamente; la validación física se realiza sobre el dispositivo.

## Licencias y trazabilidad

- Código propio de integración: MIT.
- Pesos Qwen preentrenados: Apache 2.0, autores originales identificados.
- Pesos de MJON Original: producidos por el entrenamiento independiente de MJON.
- El hecho de integrar pesos de un tercero no convierte a MJON IA en autora de esos pesos ni transfiere todas las capacidades a su modelo original.
