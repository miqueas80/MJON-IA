# MJON IA · Universidad / Open-Base

## Qué es

**MJON Open-Base** es una ruta adicional de inferencia conversacional para el navegador, basada en un modelo externo de pesos abiertos:

- **Modelo original:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), licencia Apache 2.0.
- **Conversión de navegador:** [onnx-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct).
- **Biblioteca de ejecución:** [Transformers.js v3.8.1](https://huggingface.co/docs/transformers.js/v3.8.1/en/pipelines), ejecutada en un Web Worker; el paquete se importa bajo demanda.
- **Aceleración:** WebGPU; cuantizaciones opcionales q4f16 (~483 MB) y q4 (~786 MB) además de archivos auxiliares.
- **Identidad:** variante *derivada de Qwen* y atribuida como tal; **NO es el Transformer propio entrenado desde cero**.
- **Hosting:** mismo GitHub Pages estático, sin servidor ni API para generar texto.

La ruta está en [open-base.html](../docs/open-base.html). Los modelos v0.1–v0.8 de **MJON Original** se mantienen intactos en la página principal y el repositorio usa solamente la rama `main`.

## Seguridad, privacidad y operación

El botón «Descargar y ejecutar» es el único punto que inicia la carga de la biblioteca y la descarga de pesos. No se descargan los ~0,5–0,8 GB al abrir el sitio principal ni al visualizar el aula. No se envían preguntas a una API de inferencia: tras cargar, el cálculo ocurre en el navegador.

Hugging Face y el CDN sirven los recursos iniciales, por lo que la **primera carga requiere Internet** y estos proveedores pueden observar solicitudes de descarga. Los datos del chat permanecen en memoria de la pestaña; la aplicación no los transmite ni almacena deliberadamente. El navegador puede persistir los pesos en su caché, pero el **offline no se garantiza**: deben estar disponibles todos los archivos auxiliares y pesos, y el navegador puede desalojarlos. El sitio declara esta limitación.

Las salidas del modelo no deben considerarse verificadas, especialmente en usos científicos, médicos, legales o de seguridad.

## Compatibilidad y recursos

WebGPU es requisito: no se proporciona automáticamente un modo CPU por ser demasiado lento para este perfil en dispositivos modestos. En Android, probá Chrome actualizado; tener Snapdragon 8 Gen 1 **no garantiza** que esa versión del navegador soporte todas las operaciones ni que alcance la memoria disponible. Si el perfil q4f16 falla, liberá recursos y probá q4, asumiendo mayor descarga. El usuario inicia cada intento.

La cancelación termina el Worker y libera los recursos de esa sesión. Los pesos pueden continuar en el caché hasta que el navegador los descarte. El historial se conserva únicamente en la pestaña actual.

## Examen de masterclass

El aula ofrece ejercicios de Biología, Matemática, Química, Programación y Pensamiento crítico. El modelo genera libremente la respuesta; **no hay respuestas preprogramadas**. El usuario puede marcar una respuesta como «útil» o «incorrecta» y exportar sus evaluaciones como JSON local, sin puntuaciones inventadas ni corrección automática.

Un acierto del modelo Open-Base **no cuenta como conocimiento transferido** a MJON Original. El verdadero entrenamiento entre modelos continúa en la vía de destilación v0.8: el alumno solo adquiere lo que aprende y demuestra en pruebas independientes.

## Verificación

```bash
node --check docs/web/open-base.mjs
node --check docs/web/open-base-worker.mjs
node scripts/verify_open_base.mjs
```

Estos controles validan archivos, seguridad básica de activación voluntaria y compatibilidad de mensajes. **No sustituyen una prueba manual en navegador con WebGPU real y la descarga del modelo.** Para comprobar el funcionamiento completo se necesita un dispositivo y una sesión gráfica con memoria suficiente.

## Licencias

El código de integración MJON es MIT. El modelo Qwen original usa Apache 2.0 según la ficha publicada; la conversión ONNX mantiene su procedencia. La licencia del código **no convierte a MJON en el creador de los pesos preentrenados**.
