# Corpus sintético MJON IA v0.3

Este proyecto genera datos sintéticos de demostración con:

`python -m mjon_ia.synthetic --output data/synthetic_es.txt --examples 1800 --seed 42`

- **Origen:** oraciones originales producidas mediante plantillas explícitas del propio proyecto.
- **Licencia:** CC0-1.0 para el contenido generado.
- **Idioma:** español.
- **Temática:** saludos, aritmética elemental y construcciones simples; datos deliberadamente limitados.
- **Privacidad:** ningún dato personal real o acceso a cuentas.
- **Reproducibilidad:** misma semilla y versión del generador producen los mismos bytes.
- **Partición:** se eliminan duplicados exactos por diálogo, se separa entrenamiento/validación y solo después se entrena el BPE sobre entrenamiento.
- **Riesgos:** plantillas repetitivas pueden inflar la aparente calidad de validación. No proporciona comprensión general ni un chatbot fiable.
- **Publicación:** la fuente del generador es abierta; `data/synthetic_es.txt` se crea en CI y se excluye del repositorio para evitar datos duplicados.

Un corpus futuro con textos externos debe acompañarse de fuente, autorización y compatibilidad de licencia. No entrenar automáticamente con contenido protegido, privado o con procedencia desconocida.
