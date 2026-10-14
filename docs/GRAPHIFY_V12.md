# MJON Original v1.2 — Graphify como profesor estructural de las neuronas

## Qué integra realmente

Se usa el proyecto público [Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify) en su paquete oficial `graphifyy==0.9.90`, licencia Apache 2.0, copyright y autores de Graphify. Graphify no es un cerebro neuronal: analiza código mediante AST y genera **nodos y aristas con procedencia**.

Nuestro objetivo es **aprender patrones de relaciones**, mediante actualización de pesos del Transformer propio de MJON. No se copia código de Graphify ni se sustituye el modelo. La red neuronal de MJON conserva su arquitectura, tokenizer y motor web originales.

## Extracción reproducible y local

En el ejecutor CPU de GitHub Actions:

```bash
python -m pip install "graphifyy==0.9.90"
graphify extract src/mjon_ia --code-only --no-cluster --out graphify-out/v12
python scripts/graphify_curriculum_v12.py \
  --source graphify-out/v12/graph.json --out data/graphify_v12
```

El `--code-only` es esencial: Graphify procesa código localmente mediante tree-sitter y no necesita un proveedor de LLM para los datos que usa esta etapa. **No se instala ni se importa Graphify en GitHub Pages**, y nunca recibe preguntas de usuarios. Los archivos `graphify-out/v12` y `data/graphify_v12` son artefactos regenerables.

Solo se aceptan enlaces con confianza literal `EXTRACTED` y relaciones conocidas como `calls`, `imports`, `inherits`, `contains` y `method`. Se excluyen nodos ambiguos, externos y etiquetas excesivamente largas. Un `manifest.json` registra SHA256 del grafo de origen, versión, licencia, tipos de enlaces y cantidades.

## ¿Cómo llega a las neuronas?

Graphify describe relaciones del propio código fuente de MJON. Cada enlace genera un ejemplo supervisado del tipo «¿A llama a B?» con respuesta «Sí». Otros ejemplos presentan combinaciones que no aparecen como arista extraída y enseñan «No», **siempre entendido como «no se registró en el mapa», no como certeza absoluta de inexistencia**.

Los pares de enlaces se dividen de manera determinista en **entrenamiento, desarrollo y examen final**. Se reservan las aristas de examen, no solo sus diferentes redacciones. El entrenamiento convierte las respuestas en pérdida cross-entropy y actualiza los parámetros de la red existente por backpropagation. No hay una tabla de respuestas ni recorridos de grafo disponibles en el navegador.

MJON v1.2 también practica el currículo previo, la memoria contextual y los ejercicios escolares, con regularización KL respecto a una copia congelada de v1.1. Solo se promociona un checkpoint si mejora en relaciones reservadas de desarrollo sin sacrificar aciertos escolares o de memoria de calibración, ni aumentar significativamente la pérdida del currículo antiguo.

## Cómo distinguir aprendizaje de imitación

Se publica `docs/web/graphify-v12.json` con resultados de preguntas finales sobre aristas nunca entrenadas. La predicción se genera con el Transformer MJON, sin recuperar el grafo de Graphify. Los resultados se comparan con v1.1. Si la selección conserva el paso 0, **no hubo mejora comprobada**.

El modelo no se convierte en una red neuronal gráfica (GNN), y aprender relaciones de un repositorio pequeño no demuestra comprensión general, razonamiento multietapa ni conocimiento científico nuevo. Para lograr eso harían falta grafos de contenidos validados, arquitectura y escala de datos superiores, exámenes más diversos y mejores capacidades del modelo.

## Recursos, procedencia y licencia

La dependencia Graphify solo existe durante la compilación y está fijada por versión. Los pesos de MJON y el código propio continúan con sus licencias correspondientes. La documentación de Graphify y el paquete mantienen su autoría y su licencia Apache 2.0; no se redistribuye el código de Graphify dentro de los módulos de la app.

El pipeline existente se mantiene offline-first para el usuario: los pesos MJON se sirven desde GitHub Pages y se ejecutan en JavaScript. El acceso a Internet durante Actions solo se necesita para instalar la herramienta de construcción, **no para responder conversaciones**.
