# MJON IA · Entrega funcional verificada

## Qué funciona hoy (una sola MJON)

El modo por defecto de la aplicación principal ofrece un **asistente local de alcance delimitado** con:

1. Consulta de **120 conceptos breves originales** de diez áreas. Las fuentes son fichas del currículo \`src/mjon_ia/knowledge_v11.py\`, convertidas a \`docs/web/knowledge-core.json\`.
2. Reformulación básica de la misma pregunta, definición de términos y comparación de dos conceptos encontrados en las fichas.
3. Ejemplos verificados de conceptos seleccionados y continuidad de preguntas cortas en la sesión.
4. Cálculo aritmético seguro (+, −, ×, ÷, paréntesis, porcentajes). Utiliza un intérprete de expresiones restringido, **sin eval ni ejecución de código arbitrario**.
5. Saludos, ayuda, memoria breve de nombre **solo en RAM** y respuestas prudentes cuando faltan datos.
6. Caché del sitio y de las fichas mediante Service Worker para uso sin conexión después de la primera carga correcta. Si el navegador elimina la caché, habrá que conectarse para recargar.
7. Acceso a los **modelos neuronales MJON Original históricos** desde el mismo panel, desactivando «Respuestas verificables». Esa generación sigue siendo experimental.

**Importante:** las definiciones y la aritmética son respuestas basadas en conocimiento local y algoritmos, **no aprendizaje en los pesos de MJON**. No deben presentarse como tal. El Transformer original y su entrenamiento escalan por otra línea dentro del mismo proyecto, pero aún no demuestran inteligencia general. No se importa un LLM externo.

## Examen reproducible

\`\`\`bash
node --check docs/web/grounding.mjs
node --check docs/web/app.mjs
node scripts/test_grounding.mjs
\`\`\`

Comprueba las 108 definiciones de materias (las otras 12 fichas son límites epistémicos), siete cálculos, división por cero, comparación de conceptos, contexto inmediato, privacidad, incertidumbre, integridad del corpus y conexiones del modo web. Estos son **tests de software**, no evaluación de inteligencia general.

## Publicación preservando los pesos originales

El pipeline \`.github/workflows/pages-fast.yml\` toma el artefacto \`github-pages\` de la **última compilación neuronal histórica completada con éxito**; verifica que contiene catálogo y pesos; sobrepone la interfaz actual y añade el modo local. Así no reconstruye doce modelos en cada cambio de UI y **conserva los pesos históricos** en la versión pública.

El entrenamiento neuronal íntegro queda disponible mediante \`workflow_dispatch\` en \`.github/workflows/pages.yml\`. Cuando termina con éxito, un nuevo artefacto contiene las versiones entrenadas y se puede volver a publicar la interfaz. Este flujo separa entrega de producto y costosas iteraciones de entrenamiento.

Si falta un artefacto histórico, el flujo rápido falla explícitamente y **no publica un sitio sin pesos neuronales**.

## Pruebas manuales de aceptación

En \`https://miqueas80.github.io/MJON-IA/\`, con modo verificable activado, comprobar:
- «¿Qué es una célula?» → definición y fuente.
- «¿Diferencia entre átomo y molécula?» → ambas fichas, sin inventar.
- «Calculá (18 + 6) / 3» → 8.
- «¿Cuánto es el 15% de 240?» → 36.
- «Me llamo Alex» → «¿Cómo me llamo?» → nombre en memoria de la pestaña.
- «¿Qué color tiene mi mochila?» → reconoce que no posee evidencia.
- Reabrir la página sin Internet **después de completar la primera visita online**, comprobando que el navegador realmente haya instalado el Service Worker.

## Limitaciones sin maquillaje

- El modo verificable conoce un conjunto pequeño de hechos y no sostiene conversaciones abiertas sobre cualquier tema.
- La recuperación local no demuestra razonamiento neuronal; una respuesta de definición no prueba que la red comprendió la materia.
- El modo neuronal no debe usarse para tomar decisiones importantes ni se puede afirmar que supera a Qwen/DeepSeek.
- El Service Worker depende de políticas de caché del navegador. No se garantiza disponibilidad si es eliminada.
- La última versión neuronal histórica que empaqueta el flujo rápido puede ser anterior al código en \`main\`; su identidad y cantidad se leen del catálogo efectivamente empaquetado.
- No se almacenan contraseñas. No ingreses información sensible.

## Próximo salto técnico

Entrenar MJON Neural Core con corpus extensos, GPU y SFT, evaluar respuestas libres y razonamiento, convertir los pesos al runtime web solo después de equivalencia numérica y validar en dispositivos reales. **Es trabajo pendiente: esta entrega no lo presenta como terminado.**
