# Publicación de MJON IA en GitHub Pages

La aplicación de demostración **ejecuta en el navegador** la inferencia del
Transformer propio, con pesos cuantizados exportados desde un entrenamiento
PyTorch. No existe servidor backend ni API de terceros.

## Configuración

1. En GitHub, cambiá a **Public** el repositorio cuando quieras revelar su código.
   El cambio de visibilidad lo decide el propietario; publicar no modifica los commits.
2. Entrá en **Settings > Pages > Build and deployment**.
3. Seleccioná **Source: GitHub Actions** (no "Deploy from a branch").
4. En **Actions**, ejecutá manualmente el workflow
   **Publicar MJON IA en GitHub Pages** usando **Run workflow**, si no comenzó
   automáticamente o si el primer despliegue falló antes de habilitar Pages.
5. Una vez que el workflow indique **Success**, comprobá
   https://miqueas80.github.io/MJON-IA/.

## Qué ejecuta el workflow

- Instala PyTorch CPU en el runner temporal de GitHub.
- Ejecuta las 4 pruebas del núcleo.
- Entrena un modelo experimental pequeño **desde pesos aleatorios**.
- Convierte su checkpoint a enteros cuantizados de 8 bits.
- Publica solamente el contenido de `docs/` como artefacto web.
- En el navegador se descarga el archivo estático `web/weights.txt` y se
  realiza la inferencia en JavaScript puro, sin enviar el texto del usuario al
  servidor de un proveedor de IA.

**Nota:** el entrenamiento ocurre en GitHub Actions al construir la web, no en
GitHub Pages ni en el teléfono. El navegador ejecuta inferencia local.

## Límites y seguridad

- Modelo diminuto: ~8.176 parámetros, entrenamiento de demostración.
- Solo 48 bytes de contexto. Resultados rudimentarios, no confiables.
- No se presenta como asistente para educación, medicina o decisiones reales.
- No se deben subir claves, documentos privados ni datos personales al
  repositorio público. El dataset `data/demo.txt` es sintético.

## Prueba de regresión local

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m mjon_ia.train --steps 800 --eval-every 200 --context 48 --d-model 16 --layers 1 --heads 2 --batch-size 16 --lr 0.001 --dropout 0 --device cpu --out checkpoints/browser.pt
python scripts/export_web.py --checkpoint checkpoints/browser.pt --out docs/web/weights.txt
python -m http.server 8000
```

Abrí http://localhost:8000/docs/ para probar el sitio local.

El modelo Python más grande de la CLI permanece disponible; el modelo web usa
una configuración reducida para ejecutarse hasta en dispositivos modestos.

## Comparador de versiones

Se construyen dos pesos **independientes**, no dos nombres para el mismo checkpoint:

| Perfil | Parámetros | Contexto | Entrenamiento |
| --- | ---: | ---: | --- |
| v0.1 mínimo | 8.176 | 48 bytes | 800 pasos, división heredada por bytes |
| v0.2 ampliado | 36.736 | 96 bytes | 900 pasos, separación de diálogos |

En el panel lateral elegí el modelo. La página descargará los pesos en ese momento y verificará su SHA-256. El entrenamiento ocurre solo en el ejecutor de GitHub Actions. Ambos modelos tienen resultados rudimentarios, no equivalen a asistentes funcionales; debido a las distintas particiones, sus pérdidas de validación tampoco son directamente comparables.

El workflow verifica el manifiesto y prueba inferencia real de ambos modelos en JavaScript antes de publicar.

## MJON IA v0.3: tercer perfil

La compilación también genera un corpus sintético CC0 con 1800 diálogos y entrena desde cero otro Transformer (aprox. 505.920 parámetros, vocabulario BPE de 512 unidades, 96 tokens de contexto, 500 pasos). La web descarga esos pesos en formato int8 y los ejecuta en el navegador con un cache KV causal. Los modelos anteriores permanecen como opciones.

El entrenamiento de este perfil es un experimento computacional, no una garantía de conversaciones coherentes. Las métricas solo comparan una parte pequeña de datos sintéticos; no miden razonamiento general. GitHub Pages sirve archivos estáticos: el entrenamiento se ejecuta en GitHub Actions.

## Nueva v0.5: primeras palabras

Tras entrenar v0.4, GitHub Actions lo usa como base para un ajuste supervisado de 900 pasos en tareas de lectura de atributos explícitos (color, material y estado). Publica el peso separado `weights-v05.txt` y un informe `firstwords.json` con evaluación libre frente al modelo v0.4 sobre los mismos ejemplos reservados. El navegador ofrece un modo de respuesta breve determinista para v0.5. No hay respuestas programadas en la web; el texto lo genera el Transformer.

## v0.6 — Recuerdos y correcciones

Tras construir v0.5, Actions genera un corpus sintético con turnos previos y correcciones, ajusta el propio modelo desde los pesos v0.5 y exporta `weights-v06.txt`. Se publica también `memory.json`, con resultados de 48 casos de recuerdo y corrección separados de entrenamiento, comparación v0.5/v0.6 y retención de la tarea anterior. En la web, las últimas tres parejas de mensajes se mantienen solo durante la sesión actual; la inferencia continúa en el dispositivo.

## v0.7 — Cuatro materias y transferencia

El workflow conserva v0.1–v0.6, crea 4.800 ejercicios sintéticos CC0 con `school_v07.py`, adapta los pesos de v0.6 a contexto 256, entrena la v0.7 y exporta `weights-v07.txt`. Publica además `school.json`: resultados por materia de pruebas con objetos/números nuevos y enunciados reformulados, comparación frente a v0.6 y retención de capacidades anteriores. La v0.7 usa el mismo motor de inferencia JavaScript local.

## v0.8 — Alumno con profesor abierto

El pipeline descarga `Qwen/Qwen2.5-0.5B-Instruct` únicamente **durante el build**, genera respuestas en español, registra el commit de la revisión del maestro, ajusta MJON v0.7 y exporta un octavo checkpoint cuantizado. El navegador **no descarga Qwen**, solo los pesos pequeños del alumno. Se publican `teacher-v08.json` y `distillation.json`, más referencias de licencia. Los siete modelos anteriores permanecen elegibles.

## MJON v0.9 — Conocimiento incorporado mediante entrenamiento

Actions usa Qwen solo como docente para generar respuestas, con atribución, y entrena el Transformer MJON v0.8 para generar v0.9. El sitio estático ofrece únicamente los pesos cuantizados del alumno; **no importa Transformers.js ni ejecuta Qwen**, y no contiene ningún selector de modelos externos. `teacher-v09.json` registra la procedencia del material docente y `distillation-v09.json` muestra la comparación en preguntas reservadas. No hay segundo módulo ni aplicación adicional.

## MJON Original v1.0

El entrenamiento de v1.0 usa el corpus docente creado por v0.9, una copia congelada del propio MJON v0.8 y repaso. Exporta `weights-v10.txt`, `continual-v10.json` y agrega una entrada al selector. El mismo motor JS maneja todos los checkpoints: no existe un motor externo en la página. Un paso seleccionado 0 indica que la promoción fue rechazada por calidad.

## v1.1 — Conocimiento propio

El pipeline genera un dataset original de conceptos elementales con `python -m mjon_ia.knowledge_v11`, reentrena el mismo Transformer MJON desde v1.0 y verifica la retención. Exporta `weights-v11.txt` y `knowledge-v11.json`. El navegador no descarga Qwen ni incluye modelos externos: el material es absorbido únicamente en los pesos entrenados de MJON.

## v1.2 — Graphify solo durante entrenamiento

GitHub Actions instala Graphify temporalmente, extrae grafo AST del código de MJON sin IA externa (`--code-only`), genera un corpus supervisado reproducible de relaciones verificadas y continúa entrenando el mismo Transformer. Publica el checkpoint `weights-v12.txt`, el resultado del examen `graphify-v12.json` y una ficha de procedencia `graphify-manifest-v12.json`. Ningún paquete Graphify se instala en el navegador ni se consulta el grafo para generar respuestas.
