"""Versión 7: agente con herramientas + memoria en disco + streaming + system prompt + RAG.

RAG (Retrieval-Augmented Generation) = antes de responder, el bot busca
información en TUS documentos y responde usando lo que encontró.

Cómo funciona aquí, en 3 pasos:
    1. INDEXAR: al arrancar, leemos los .txt y .md de la carpeta documentos/
       y los partimos en fragmentos: uno por sección del documento.
    2. BUSCAR: le damos a Claude una herramienta nueva, buscar_documentos.
       Cuando le preguntás algo que podría estar en tus documentos, la usa;
       nosotros devolvemos los fragmentos más parecidos a la consulta.
    3. RESPONDER: Claude lee esos fragmentos y responde citando la fuente.

La búsqueda es por palabras clave (TF-IDF), en Python puro, sin librerías
extra. Los RAG "de producción" suelen usar embeddings (vectores que capturan
el significado, no solo las palabras), pero la idea es exactamente la misma:
solo cambia la función que puntúa qué fragmentos son relevantes.

El manual de ejemplo tiene unas quince secciones y la búsqueda devuelve
las 3 más relevantes: así se ve que el bot lee solo una parte, no todo.

Para usarlo con tus propios documentos, copiá archivos .txt o .md a la
carpeta documentos/ y reiniciá el programa.

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v07.py
"""

import ast
import json
import math
import operator
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
MODEL = "claude-opus-5-5"
CARPETA = Path(__file__).parent
ARCHIVO_MEMORIA = CARPETA / "memoria_v07.json"
CARPETA_DOCUMENTOS = CARPETA / "documentos"

# System prompt: se mantienen las reglas de la v06 (calcular, obtener_hora, no
# inventar) y se suma lo nuevo de la v07: buscar en los documentos y citar la
# fuente. Si sacáramos las reglas de la v06, el bot volvería a hacer cuentas de memoria.
SYSTEM = (
    "Sos Tecla, una asistente amable y directa. Respondés siempre en español "
    "rioplatense (de vos), de forma breve y clara, sin rodeos. "
    "Para cualquier cuenta usá la herramienta calcular; nunca calcules de memoria. "
    "Para la fecha o la hora usá la herramienta obtener_hora. "
    "Tenés acceso a los documentos del usuario con la herramienta buscar_documentos. "
    "Si la pregunta puede estar respondida en esos documentos, buscala antes de "
    "responder y citá el archivo de donde sacaste la información. Si no encontrás "
    "nada relevante, decilo en lugar de inventar."
)


# ---------------------------------------------------------------------------
# RAG (lo nuevo de la v07)
# ---------------------------------------------------------------------------

# Palabras vacías: aparecen en casi cualquier texto y no dicen de qué trata.
# Si las contáramos, una sección cualquiera podría ganarle a la correcta solo
# por repetir "que" o "con". (Van sin tildes porque tokenizar() las quita
# antes de comparar.)
PALABRAS_VACIAS = set("""
    que con los las del por para una uno unos unas como mas pero sus les nos mis tus
    este esta esto estos estas ese esa eso esos esas son hay muy sin sobre ante entre
    desde hasta cuando donde quien cual cuanto cuanta cuantos cuantas ser estar fue era
    tiene tengo tienen puedo puede pueden quiero algo algun alguna cada solo otro otra
    todo todos toda todas tambien ya asi les ella ellos usted hola
""".split())


def tokenizar(texto: str) -> list[str]:
    """Pasa a minúsculas, quita tildes y separa en palabras de 3+ letras,
    sin las palabras vacías."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return [p for p in re.findall(r"\w+", texto) if len(p) > 2 and p not in PALABRAS_VACIAS]


def partir_en_fragmentos(texto: str, max_chars: int = 800) -> list[str]:
    """Parte el texto en fragmentos: uno por sección (cada título que empieza
    con #) y, si una sección es muy larga, la divide por párrafos hasta
    ~max_chars.

    Por qué por secciones: cada fragmento debería hablar de UN tema. Si
    mezcla varios, la búsqueda trae texto que no viene al caso y, con un
    documento corto, termina devolviendo casi todo el documento: ahí ya no
    se nota la diferencia entre buscar bien y buscar mal."""
    fragmentos = []
    for seccion in re.split(r"\n(?=#+ )", texto):  # corta justo antes de cada título
        actual = ""
        for parrafo in re.split(r"\n\s*\n", seccion):
            parrafo = parrafo.strip()
            if not parrafo:
                continue
            if actual and len(actual) + len(parrafo) > max_chars:
                fragmentos.append(actual)
                actual = ""
            actual = f"{actual}\n\n{parrafo}" if actual else parrafo
        if actual:
            fragmentos.append(actual)
    return fragmentos


def indexar_documentos() -> list[dict]:
    """Paso 1: lee todos los documentos y arma la lista de fragmentos."""
    indice = []
    if not CARPETA_DOCUMENTOS.exists():
        return indice
    for archivo in sorted(CARPETA_DOCUMENTOS.rglob("*")):
        if archivo.suffix.lower() not in (".txt", ".md"):
            continue
        texto = archivo.read_text(encoding="utf-8", errors="ignore")
        for fragmento in partir_en_fragmentos(texto):
            indice.append({
                "fuente": archivo.name,
                "texto": fragmento,
                "palabras": Counter(tokenizar(fragmento)),
            })
    return indice


INDICE = indexar_documentos()

# IDF: las palabras que aparecen en pocos fragmentos valen más que las comunes
_df = Counter(p for frag in INDICE for p in frag["palabras"])
IDF = {p: math.log((len(INDICE) + 1) / (n + 0.5)) for p, n in _df.items()}


def buscar_documentos(consulta: str, cantidad: int = 3) -> str:
    """Paso 2: devuelve los fragmentos más relevantes para la consulta."""
    if not INDICE:
        return "No hay documentos cargados en la carpeta documentos/."

    palabras_consulta = set(tokenizar(consulta))
    puntajes = []
    for frag in INDICE:
        puntaje = sum(
            IDF.get(p, 0) * frag["palabras"][p] / (frag["palabras"][p] + 1)
            for p in palabras_consulta
        )
        if puntaje > 0:
            puntajes.append((puntaje, frag))

    if not puntajes:
        return "No encontré nada relevante en los documentos."

    puntajes.sort(key=lambda x: x[0], reverse=True)
    return "\n\n---\n\n".join(
        f"[Fuente: {frag['fuente']}]\n{frag['texto']}" for _, frag in puntajes[:cantidad]
    )


# ---------------------------------------------------------------------------
# Herramientas de antes: reloj y calculadora
# ---------------------------------------------------------------------------

def obtener_hora() -> str:
    """Devuelve la fecha y hora actual del sistema."""
    return datetime.now().strftime("%A %d/%m/%Y, %H:%M:%S")


# Operadores permitidos en la calculadora. Evitamos eval() por seguridad:
# eval ejecutaría cualquier código que le llegue (de Claude o de un texto
# malicioso escondido en un documento).
OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Límites para que una cuenta enorme no cuelgue el programa. Sin ellos,
# '9**9**9' (un número de cientos de millones de cifras) deja a Python
# calculando durante horas y el chat no responde más.
MAX_EXPONENTE = 100
MAX_CIFRAS = 1000


def calcular(expresion: str) -> str:
    """Evalúa una expresión aritmética como '(2 + 3) * 4 / 7'."""

    def controlar(valor):
        if isinstance(valor, int) and abs(valor) >= 10 ** MAX_CIFRAS:
            raise ValueError(f"El resultado tiene más de {MAX_CIFRAS} cifras")
        return valor

    def evaluar(nodo):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
            return nodo.value
        if isinstance(nodo, ast.BinOp) and type(nodo.op) in OPERADORES:
            izquierda, derecha = evaluar(nodo.left), evaluar(nodo.right)
            if isinstance(nodo.op, ast.Pow) and abs(derecha) > MAX_EXPONENTE:
                raise ValueError(f"Exponente demasiado grande (máximo {MAX_EXPONENTE})")
            return controlar(OPERADORES[type(nodo.op)](izquierda, derecha))
        if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in OPERADORES:
            return OPERADORES[type(nodo.op)](evaluar(nodo.operand))
        raise ValueError("Expresión no permitida")

    try:
        return str(evaluar(ast.parse(expresion, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


HERRAMIENTAS = [
    {
        "name": "buscar_documentos",
        "description": "Busca información en los documentos del usuario y devuelve los "
                       "fragmentos más relevantes con su fuente. La búsqueda es por palabras "
                       "clave: usá términos concretos que probablemente aparezcan en el texto. "
                       "Podés llamarla varias veces con distintas palabras si no encontrás algo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Palabras clave a buscar, por ejemplo 'política devoluciones plazo'",
                }
            },
            "required": ["consulta"],
        },
    },
    {
        "name": "obtener_hora",
        "description": "Devuelve la fecha y hora actual. Usala cuando el usuario pregunte la hora o la fecha.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "calcular",
        "description": "Calcula una expresión aritmética. Soporta + - * / ** % y paréntesis. "
                       "Usala para cualquier cuenta en lugar de calcular de memoria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expresion": {
                    "type": "string",
                    "description": "La expresión a calcular, por ejemplo '(234 * 17) / 3'",
                }
            },
            "required": ["expresion"],
        },
    },
]


def ejecutar_herramienta(nombre: str, argumentos: dict) -> str:
    if nombre == "buscar_documentos":
        return buscar_documentos(argumentos["consulta"])
    if nombre == "obtener_hora":
        return obtener_hora()
    if nombre == "calcular":
        return calcular(argumentos["expresion"])
    return f"Herramienta desconocida: {nombre}"


# ---------------------------------------------------------------------------
# Memoria en disco
# ---------------------------------------------------------------------------

def cargar_memoria() -> list:
    """Lee la conversación guardada, o empieza vacía si no hay archivo."""
    if ARCHIVO_MEMORIA.exists():
        return json.loads(ARCHIVO_MEMORIA.read_text(encoding="utf-8"))
    return []


def guardar_memoria(memoria: list) -> None:
    """Escribe la conversación completa en el archivo JSON."""
    ARCHIVO_MEMORIA.write_text(
        json.dumps(memoria, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def a_dict(bloque) -> dict:
    """Convierte un bloque de la respuesta de Claude (objeto del SDK) en un
    diccionario simple, para poder guardarlo en JSON."""
    if bloque.type == "text":
        return {"type": "text", "text": bloque.text}
    if bloque.type == "tool_use":
        return {"type": "tool_use", "id": bloque.id, "name": bloque.name, "input": bloque.input}
    return bloque.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

memoria = cargar_memoria()

print("Chatbot v07 (herramientas + memoria + streaming + system prompt + RAG). Escribí 'salir' para terminar u 'olvidar' para borrar la memoria.")
archivos = {frag["fuente"] for frag in INDICE}
print(f"(Documentos indexados: {len(archivos)} archivos, {len(INDICE)} fragmentos)")
if memoria:
    print(f"(Memoria cargada: {len(memoria)} mensajes de sesiones anteriores)")
print()

while True:
    user_input = input("Vos: ").strip()
    if user_input.lower() in ("salir", "exit", "quit"):
        break
    if user_input.lower() == "olvidar":
        memoria.clear()
        ARCHIVO_MEMORIA.unlink(missing_ok=True)
        print("\n(Memoria borrada, también del disco)\n")
        continue
    if not user_input:
        continue

    memoria.append({"role": "user", "content": user_input})

    # Bucle del agente: seguimos mientras Claude pida herramientas
    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=HERRAMIENTAS,
            messages=memoria,
        ) as stream:
            hubo_texto = False
            for pedazo in stream.text_stream:
                if not hubo_texto:
                    print("\nBot: ", end="")
                    hubo_texto = True
                print(pedazo, end="", flush=True)  # flush: mostrarlo ya, sin esperar
            if hubo_texto:
                print()

            response = stream.get_final_message()

        memoria.append({"role": "assistant", "content": [a_dict(b) for b in response.content]})

        if response.stop_reason != "tool_use":
            break  # Claude ya tiene la respuesta final

        resultados = []
        for bloque in response.content:
            if bloque.type == "tool_use":
                resultado = ejecutar_herramienta(bloque.name, bloque.input)
                resumen = resultado if len(resultado) < 100 else resultado[:100].replace("\n", " ") + "..."
                print(f"  [herramienta] {bloque.name}({bloque.input}) -> {resumen}")
                resultados.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": resultado,
                })
        memoria.append({"role": "user", "content": resultados})

    # Guardamos solo cuando el turno terminó completo (sin herramientas pendientes)
    guardar_memoria(memoria)
    print()  # el texto ya se mostró en vivo; solo dejamos una línea en blanco
