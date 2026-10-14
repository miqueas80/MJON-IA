"""MJON Original v1.1: currículo FUNDAMENTAL escrito para entrenar sus pesos.

Material educativo introductorio original: preguntas y respuestas redactadas
para MJON IA, no copiadas de material escolar ni descargadas de una API.
Licencia CC0 de la redacción propia. No implica comprensión general.
"""
import argparse
import json
import random
from pathlib import Path

# subject | concept | short verified answer
# Cada dominio reserva los dos últimos conceptos para evaluación de conceptos
# desconocidos; los diez primeros son enseñados al alumno.
RAW = """
biologia|célula|La unidad básica estructural y funcional de los seres vivos.
biologia|ADN|La molécula que almacena información genética hereditaria.
biologia|mitosis|División celular que produce dos células hijas equivalentes.
biologia|fotosíntesis|Proceso por el que organismos usan luz para producir sustancias orgánicas.
biologia|respiración celular|Conjunto de reacciones que permite obtener energía química.
biologia|tejido biológico|Conjunto organizado de células con funciones relacionadas.
biologia|órgano|Estructura formada por tejidos que realizan funciones específicas.
biologia|ecosistema|Conjunto de seres vivos y su ambiente en interacción.
biologia|evolución biológica|Cambio de características hereditarias en poblaciones a lo largo de generaciones.
biologia|homeostasis|Mantenimiento de condiciones internas relativamente estables.
biologia|enzima|Molécula que acelera reacciones químicas biológicas.
biologia|mutación genética|Cambio en la secuencia del material genético.
quimica|átomo|Unidad de un elemento químico formada por núcleo y electrones.
quimica|protón|Partícula subatómica con carga eléctrica positiva.
quimica|electrón|Partícula subatómica con carga eléctrica negativa.
quimica|neutrón|Partícula subatómica sin carga eléctrica neta.
quimica|molécula|Conjunto definido de átomos unidos por enlaces químicos.
quimica|elemento químico|Sustancia cuyos átomos tienen igual número de protones.
quimica|compuesto químico|Sustancia formada por elementos distintos en proporción definida.
quimica|reacción química|Transformación de sustancias mediante reorganización de enlaces.
quimica|disolución|Mezcla homogénea de soluto y disolvente.
quimica|pH|Medida relacionada con la actividad de los iones hidrógeno.
quimica|enlace covalente|Enlace químico en el que se comparten electrones.
quimica|oxidación|Proceso químico que implica pérdida de electrones.
fisica|fuerza|Interacción capaz de cambiar el movimiento o deformar un cuerpo.
fisica|masa|Magnitud que cuantifica la inercia de un cuerpo.
fisica|aceleración|Cambio de velocidad por unidad de tiempo.
fisica|velocidad|Variación de posición respecto del tiempo, con dirección y sentido.
fisica|energía cinética|Energía asociada al movimiento de un cuerpo.
fisica|trabajo mecánico|Transferencia de energía por una fuerza durante un desplazamiento.
fisica|potencia|Cantidad de trabajo o energía transferida por unidad de tiempo.
fisica|densidad|Relación entre la masa y el volumen de un material.
fisica|temperatura|Magnitud relacionada con el estado térmico de un sistema.
fisica|corriente eléctrica|Flujo neto de carga eléctrica por unidad de tiempo.
fisica|resistencia eléctrica|Medida de oposición al paso de corriente eléctrica.
fisica|presión|Fuerza normal ejercida por unidad de superficie.
matematica|fracción|Expresión que representa partes o cocientes mediante numerador y denominador.
matematica|porcentaje|Proporción expresada sobre una base de cien.
matematica|número primo|Entero mayor que uno con exactamente dos divisores positivos.
matematica|número par|Entero divisible entre dos sin resto.
matematica|ecuación|Igualdad matemática que contiene una o más incógnitas.
matematica|variable|Símbolo que representa un valor que puede cambiar.
matematica|función matemática|Correspondencia que asigna a cada entrada una única salida.
matematica|perímetro|Longitud total del contorno de una figura.
matematica|área|Medida de la superficie de una figura.
matematica|media aritmética|Suma de los valores dividida por su cantidad.
matematica|probabilidad|Medida de la posibilidad de ocurrencia de un evento.
matematica|valor absoluto|Distancia de un número al cero en la recta real.
lengua|sustantivo|Palabra que nombra entidades, seres, lugares o conceptos.
lengua|verbo|Palabra que expresa acciones, procesos o estados.
lengua|adjetivo|Palabra que describe o califica al sustantivo.
lengua|adverbio|Palabra que modifica un verbo, adjetivo u otro adverbio.
lengua|sinónimo|Palabra con significado igual o semejante a otra.
lengua|antónimo|Palabra con sentido opuesto al de otra.
lengua|párrafo|Conjunto de oraciones que desarrolla una unidad temática.
lengua|idea principal|Contenido central que organiza el sentido de un texto.
lengua|metáfora|Recurso que presenta una cosa mediante otra por semejanza.
lengua|argumento|Razón o conjunto de razones para sostener una afirmación.
lengua|cohesión textual|Relación entre partes del texto mediante recursos lingüísticos.
lengua|tesis argumentativa|Afirmación principal que un texto intenta defender.
historia|fuente primaria|Testimonio o documento producido en la época estudiada.
historia|fuente secundaria|Interpretación elaborada a partir de fuentes previas.
historia|cronología|Organización de acontecimientos según su secuencia temporal.
historia|revolución industrial|Transformación productiva impulsada por mecanización y fábricas.
historia|independencia política|Condición de un territorio que no depende de otro Estado.
historia|democracia|Sistema político que reconoce participación y soberanía popular.
historia|constitución|Norma fundamental que organiza el Estado y reconoce derechos.
historia|ciudadanía|Condición de pertenencia política con derechos y responsabilidades.
historia|derechos humanos|Derechos inherentes a las personas por su dignidad.
historia|patrimonio cultural|Bienes y expresiones que una comunidad valora y transmite.
historia|reforma social|Cambio deliberado en instituciones o condiciones colectivas.
historia|historiografía|Estudio de cómo se investiga y escribe la historia.
geografia|latitud|Distancia angular respecto del ecuador expresada en grados.
geografia|longitud geográfica|Distancia angular respecto del meridiano de referencia.
geografia|clima|Patrón de condiciones atmosféricas durante períodos prolongados.
geografia|tiempo atmosférico|Estado de la atmósfera en un lugar y momento.
geografia|cuenca hidrográfica|Territorio cuyas aguas drenan hacia un mismo punto.
geografia|bioma|Región con comunidades biológicas vinculadas al clima y ambiente.
geografia|recurso renovable|Recurso que puede reponerse naturalmente bajo ciertas condiciones.
geografia|urbanización|Proceso de crecimiento de población y espacios urbanos.
geografia|migración|Desplazamiento de personas entre lugares de residencia.
geografia|escala cartográfica|Relación entre distancia en un mapa y distancia real.
geografia|relieve terrestre|Conjunto de formas y diferencias de altura del terreno.
geografia|densidad demográfica|Habitantes por unidad de superficie en una región.
informatica|algoritmo|Secuencia finita y ordenada de pasos para resolver un problema.
informatica|variable informática|Espacio lógico asociado a un dato que puede cambiar.
informatica|función informática|Bloque reutilizable de instrucciones con entradas y posibles salidas.
informatica|condición lógica|Expresión que determina qué instrucciones se ejecutan.
informatica|bucle|Estructura que repite instrucciones según una condición.
informatica|base de datos|Conjunto organizado de datos que permite consultarlos.
informatica|sistema operativo|Software que administra recursos y ofrece servicios a programas.
informatica|memoria RAM|Memoria de trabajo volátil utilizada por programas en ejecución.
informatica|código abierto|Software cuyo código fuente puede examinarse y modificarse según su licencia.
informatica|cifrado|Transformación de información mediante una clave para proteger su confidencialidad.
informatica|copia de seguridad|Duplicado de datos creado para recuperarlos ante una pérdida.
informatica|red local|Conjunto de dispositivos conectados en un área limitada.
pensamiento|premisa|Proposición utilizada como punto de partida de un argumento.
pensamiento|conclusión|Proposición que se desprende de las premisas de un razonamiento.
pensamiento|evidencia|Información que permite apoyar o cuestionar una afirmación.
pensamiento|hipótesis|Explicación provisional que puede ponerse a prueba.
pensamiento|experimento controlado|Prueba que compara condiciones manteniendo otras variables controladas.
pensamiento|correlación|Relación estadística entre variables que no demuestra causalidad.
pensamiento|causalidad|Relación en la que un factor contribuye a producir un efecto.
pensamiento|sesgo de confirmación|Tendencia a favorecer información que apoya creencias previas.
pensamiento|fuente confiable|Fuente cuya procedencia y evidencia pueden comprobarse.
pensamiento|incertidumbre|Situación en la que no se conocen completamente los resultados.
pensamiento|pensamiento crítico|Evaluación de afirmaciones mediante lógica y evidencia.
pensamiento|reproducibilidad|Capacidad de obtener resultados coherentes al repetir un procedimiento.
limites|color de mi mochila|No tengo esa información.
limites|mi contraseña secreta|No conozco tu contraseña.
limites|contenido de mi escritorio|No puedo verlo sin información.
limites|clima exacto de mañana|Necesito datos meteorológicos actualizados.
limites|lo que pienso ahora|No puedo leer tus pensamientos.
limites|ubicación de mis llaves|No sé dónde están tus llaves.
limites|edad de una persona desconocida|No tengo datos para determinarla.
limites|resultado de una prueba no realizada|No se puede saber sin hacer la prueba.
limites|identidad de una foto no compartida|No tengo acceso a esa foto.
limites|precios actuales de una tienda|Necesito consultar precios actualizados.
limites|nombre de mi mascota sin decirlo|No conozco ese dato.
limites|contenido de un documento no entregado|No puedo saberlo sin ver el documento.
""".strip()

VARIANTS = (
    "¿Qué significa {concept}?",
    "Explicá brevemente {concept}.",
    "Definí {concept} en una frase.",
    "¿Qué entendés por {concept}?",
    "En palabras sencillas, ¿qué es {concept}?",
    "¿Cómo describirías {concept} a un estudiante?",
)
HOLDOUT_VARIANT = "En una respuesta corta, explicá el significado de {concept}."
LIMIT_VARIANTS = (
    "¿Sabés {concept}?",
    "¿Podés decirme {concept}?",
    "¿Conocés {concept}?",
    "¿Tenés información sobre {concept}?",
    "¿Podés saber {concept} sin ningún dato adicional?",
)
LIMIT_HOLDOUT = "Sin inventar información, decime {concept}."


def parse(raw=RAW):
    data={}
    for index,line in enumerate(raw.splitlines()):
        cells=line.split("|")
        if len(cells)!=3:
            raise ValueError(f"Entrada curricular inválida en línea {index+1}")
        subject,concept,answer=map(str.strip,cells)
        if not (subject and concept and answer):
            raise ValueError("Hay una entrada vacía")
        data.setdefault(subject,[]).append((concept,answer))
    if len(data)<10 or any(len(rows)!=12 for rows in data.values()):
        raise ValueError("Se requieren diez áreas con doce conceptos cada una")
    if len({(subject,concept) for subject,rows in data.items() for concept,_ in rows})!=120:
        raise ValueError("Hay conceptos repetidos")
    return data


def build(seed=1111,train_variants=4):
    if not 1<=train_variants<=5:
        raise ValueError("Número de reformulaciones inválido")
    train,paraphrase,unknown=[],[],[]
    for subject,entries in parse().items():
        for i,(concept,answer) in enumerate(entries):
            key=f"{subject}:{i}"
            row={"subject":subject,"concept":key,"answer":answer}
            patterns = LIMIT_VARIANTS if subject=="limites" else VARIANTS
            heldout_pattern = LIMIT_HOLDOUT if subject=="limites" else HOLDOUT_VARIANT
            if i<10:
                for pattern in patterns[:train_variants]:
                    train.append({**row,"question":pattern.format(concept=concept)})
                paraphrase.append({**row,"question":heldout_pattern.format(concept=concept)})
            else:
                for pattern in (patterns[0],heldout_pattern):
                    unknown.append({**row,"question":pattern.format(concept=concept)})
    random.Random(seed).shuffle(train)
    random.Random(seed+1).shuffle(paraphrase)
    random.Random(seed+2).shuffle(unknown)
    seen={r["concept"] for r in train}
    if seen & {r["concept"] for r in unknown}:
        raise ValueError("Fuga de conceptos reservados")
    if {r["question"] for r in train} & {r["question"] for r in paraphrase+unknown}:
        raise ValueError("Fuga de formulaciones reservadas")
    return train,paraphrase,unknown


def save(out="data/knowledge_v11",seed=1111):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    collections=dict(zip(("train","paraphrase","unknown"),build(seed)))
    for name,records in collections.items():
        (folder/(name+".json")).write_text(
            json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MJON Conocimiento v1.1:",{key:len(rows) for key,rows in collections.items()},
          "conceptos que se enseñan: 100; conceptos nuevos: 20",flush=True)
    return collections


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="data/knowledge_v11")
    p.add_argument("--seed",type=int,default=1111)
    a=p.parse_args()
    save(a.out,a.seed)
