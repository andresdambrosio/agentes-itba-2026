"""Versión 10: v09 + prompt caching.

Qué agrega respecto de la v09 y por qué:
En cada llamada reenviamos TODO: herramientas, system prompt y el historial
completo. Casi todo es igual a la llamada anterior; solo cambia el final.
Pagar una y otra vez por procesar lo mismo es caro y lento.

    - Con cache_control={"type": "ephemeral"} le pedimos a la API que guarde
      ese prefijo. En la llamada siguiente lo lee de la caché: esos tokens se
      cobran a una fracción del precio normal y la respuesta llega antes.
    - El bot muestra "[caché] N tokens leídos de la caché" cada vez que lo
      aprovecha, y el registro de la v09 muestra cómo baja el costo del turno.

Ojo: la caché solo sirve si el prefijo es idéntico. Si el system prompt o
las herramientas cambian a mitad de la charla, se pierde. Por eso, más
adelante, la v14 y la v16 arman el system prompt una sola vez al arrancar.
La caché también vence si pasan unos minutos sin usarla.

Uso:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v10.py
"""

import ast
import json
import math
import operator
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic

# max_retries (de la v08): el SDK reintenta solo los errores de red, 429 y 5xx
client = anthropic.Anthropic(max_retries=4)
MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
CARPETA = Path(__file__).parent
ARCHIVO_MEMORIA = CARPETA / "memoria_v10.json"
ARCHIVO_REGISTRO = CARPETA / "registro.jsonl"     # v09: una línea por llamada y por herramienta
CARPETA_DOCUMENTOS = CARPETA / "documentos"

MAX_VUELTAS = 10           # tope de llamadas seguidas a herramientas por turno (v08)

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
# RAG (de la v07)
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
    """Paso 1 (INDEXAR): lee todos los documentos y arma la lista de fragmentos."""
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
    """Devuelve los fragmentos más relevantes para la consulta."""
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
    raise ValueError(f"Herramienta desconocida: {nombre}")


# ---------------------------------------------------------------------------
# Memoria en disco
# ---------------------------------------------------------------------------

def cargar_memoria() -> list:
    """Lee la conversación guardada. Si el archivo está dañado, lo aparta y
    empieza de cero en lugar de caerse (robustez)."""
    if not ARCHIVO_MEMORIA.exists():
        return []
    try:
        return json.loads(ARCHIVO_MEMORIA.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        respaldo = ARCHIVO_MEMORIA.with_suffix(".danado.json")
        ARCHIVO_MEMORIA.rename(respaldo)
        print(f"(La memoria estaba dañada; la moví a {respaldo.name} y empiezo de cero)")
        return []


def guardar_memoria(memoria: list) -> None:
    ARCHIVO_MEMORIA.write_text(
        json.dumps(memoria, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def a_dict(bloque) -> dict:
    """Convierte un bloque de la respuesta (objeto del SDK) en un diccionario."""
    if bloque.type == "text":
        return {"type": "text", "text": bloque.text}
    if bloque.type == "tool_use":
        return {"type": "tool_use", "id": bloque.id, "name": bloque.name, "input": bloque.input}
    # Otros bloques (por ejemplo "thinking") se guardan tal cual llegaron
    return bloque.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Registro de cada paso (de la v09)
# ---------------------------------------------------------------------------

# Precios de lista en dólares por millón de tokens: (entrada, salida).
# Fuente: documentación de Anthropic, noviembre de 2026
# (https://platform.claude.com/docs/en/about-claude/pricing). Los precios
# cambian: revisalos antes de sacar conclusiones. Haiku tiene precio
# escalonado según el largo del prompt; acá va el más bajo.
PRECIOS = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Los tokens de la caché (v10) se cobran distinto: escribirlos cuesta 1,25
# veces la entrada normal y leerlos, 0,1 veces.
ESCRIBIR_CACHE, LEER_CACHE = 1.25, 0.10

# Lo que va sumando el turno en curso, para el resumen que se muestra al final
turno_actual = {"llamadas": 0, "herramientas": 0, "tokens": 0, "costo": 0.0, "inicio": 0.0}


def registrar(evento: dict) -> None:
    """Agrega una línea JSON a registro.jsonl. Un archivo con un evento por
    línea se puede leer, filtrar y sumar después (con Python, una planilla o
    una herramienta de observabilidad)."""
    evento = {"momento": datetime.now().isoformat(timespec="seconds"),
              "programa": Path(__file__).stem, **evento}
    with open(ARCHIVO_REGISTRO, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def costo_estimado(modelo: str, uso) -> float:
    """Costo en dólares de una llamada, según los tokens que informa la API."""
    precio = next((p for nombre, p in PRECIOS.items() if modelo.startswith(nombre)), (0.0, 0.0))
    entrada, salida = precio
    return (uso.input_tokens * entrada
            + (uso.cache_creation_input_tokens or 0) * entrada * ESCRIBIR_CACHE
            + (uso.cache_read_input_tokens or 0) * entrada * LEER_CACHE
            + uso.output_tokens * salida) / 1_000_000


def registrar_llamada(response, inicio: float) -> None:
    """Anota una llamada al modelo: cuánto tardó, cuántos tokens y cuánto costó."""
    uso = response.usage
    tokens = (uso.input_tokens + (uso.cache_creation_input_tokens or 0)
              + (uso.cache_read_input_tokens or 0) + uso.output_tokens)
    costo = costo_estimado(response.model, uso)
    registrar({
        "tipo": "llamada", "modelo": response.model, "stop_reason": response.stop_reason,
        "segundos": round(time.monotonic() - inicio, 2),
        "tokens_entrada": uso.input_tokens,
        "tokens_cache_escritos": uso.cache_creation_input_tokens or 0,
        "tokens_cache_leidos": uso.cache_read_input_tokens or 0,
        "tokens_salida": uso.output_tokens,
        "costo_usd": round(costo, 6),
    })
    turno_actual["llamadas"] += 1
    turno_actual["tokens"] += tokens
    turno_actual["costo"] += costo


def registrar_herramienta(bloque, resultado: str, es_error: bool, inicio: float) -> None:
    """Anota una herramienta: con qué argumentos se llamó, qué devolvió y si falló."""
    registrar({
        "tipo": "herramienta", "nombre": bloque.name,
        "argumentos": json.dumps(bloque.input, ensure_ascii=False)[:300],
        "segundos": round(time.monotonic() - inicio, 2),
        "error": es_error, "resultado": resultado[:300],
    })
    turno_actual["herramientas"] += 1


def empezar_registro_de_turno() -> None:
    turno_actual.update(llamadas=0, herramientas=0, tokens=0, costo=0.0, inicio=time.monotonic())


def resumen_del_turno() -> str:
    t = turno_actual
    tokens = f"{t['tokens']:,}".replace(",", ".")
    costo = f"{t['costo']:.4f}" if t["costo"] >= 0.001 else f"{t['costo']:.6f}"  # Haiku cuesta muy poco
    return (f"  [registro] {t['llamadas']} llamadas, {t['herramientas']} herramientas, "
            f"{tokens} tokens, {time.monotonic() - t['inicio']:.1f} s, ~US$ {costo}")


# ---------------------------------------------------------------------------
# Una llamada a Claude, con streaming
# ---------------------------------------------------------------------------

def llamar_claude(memoria: list, ultima_vuelta: bool = False, mostrar: bool = True):
    """Hace una llamada con streaming y devuelve el mensaje completo."""
    inicio = time.monotonic()  # registro (de la v09): para medir cuánto tarda
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        tools=HERRAMIENTAS,
        # En la última vuelta permitida, Claude ya no puede pedir herramientas
        tool_choice={"type": "none"} if ultima_vuelta else {"type": "auto"},
        # Prompt caching (lo nuevo de la v10): guarda el prefijo (herramientas +
        # system + mensajes) para que la próxima llamada lo lea de la caché.
        cache_control={"type": "ephemeral"},
        messages=memoria,
    ) as stream:
        hubo_texto = False
        for pedazo in stream.text_stream:
            if mostrar:
                if not hubo_texto:
                    print("\nBot: ", end="")
                print(pedazo, end="", flush=True)  # flush: mostrarlo ya, sin esperar
            hubo_texto = True
        if mostrar and hubo_texto:
            print()
        response = stream.get_final_message()
    registrar_llamada(response, inicio)  # de la v09
    return response


# ---------------------------------------------------------------------------
# Un turno completo del agente (de la v08)
# ---------------------------------------------------------------------------

def turno(memoria: list, texto: str, mostrar: bool = True) -> str:
    """Agrega el mensaje del usuario, corre el bucle del agente y devuelve el
    texto final. Si algo falla, deja la memoria como estaba antes del turno."""
    inicio = len(memoria)
    memoria.append({"role": "user", "content": texto})
    try:
        return _bucle_agente(memoria, mostrar)
    except BaseException:
        del memoria[inicio:]  # deshacer el turno: nada de mensajes huérfanos
        raise


def _bucle_agente(memoria: list, mostrar: bool) -> str:
    for vuelta in range(MAX_VUELTAS):
        ultima = vuelta == MAX_VUELTAS - 1
        response = llamar_claude(memoria, ultima_vuelta=ultima, mostrar=mostrar)
        # Lo nuevo de la v10: mostramos cuánto se aprovechó la caché
        if mostrar and response.usage.cache_read_input_tokens:
            print(f"  [caché] {response.usage.cache_read_input_tokens} tokens leídos de la caché")

        contenido = [a_dict(b) for b in response.content]

        if response.stop_reason == "max_tokens":
            # Una herramienta cortada a la mitad no se puede ejecutar ni
            # dejar sin resultado: la sacamos y avisamos.
            contenido = [b for b in contenido if b["type"] != "tool_use"] or [
                {"type": "text", "text": "(respuesta vacía)"}]
            memoria.append({"role": "assistant", "content": contenido})
            if mostrar:
                print("  [aviso] La respuesta se cortó por max_tokens.")
            return texto_de(response)

        if response.stop_reason == "refusal":
            # Claude no quiso responder: no guardamos el turno en la memoria
            raise RespuestaRechazada()

        memoria.append({"role": "assistant", "content": contenido})

        if response.stop_reason != "tool_use":
            return texto_de(response)

        resultados = []
        for bloque in response.content:
            if bloque.type == "tool_use":
                resultados.append(resultado_herramienta(bloque, mostrar))
        if vuelta == MAX_VUELTAS - 2:
            # Avisamos a Claude que la próxima es su última oportunidad
            resultados.append({"type": "text", "text": "Llegaste al límite de herramientas: "
                               "respondé ahora con la información que tenés."})
        memoria.append({"role": "user", "content": resultados})

    return ""  # no se llega aquí: la última vuelta no permite herramientas


class RespuestaRechazada(Exception):
    """Claude terminó con stop_reason == "refusal"."""


def resultado_herramienta(bloque, mostrar: bool) -> dict:
    """Ejecuta una herramienta. Si falla, Claude recibe el error (is_error)
    en lugar de que se caiga el programa."""
    inicio = time.monotonic()
    try:
        resultado = ejecutar_herramienta(bloque.name, bloque.input)
        es_error = False
    except Exception as e:
        resultado, es_error = f"Error: {e}", True
    registrar_herramienta(bloque, resultado, es_error, inicio)  # de la v09
    if mostrar:
        resumen = resultado if len(resultado) < 100 else resultado[:100].replace("\n", " ") + "..."
        print(f"  [herramienta] {bloque.name}({bloque.input}) -> {resumen}")
    return {"type": "tool_result", "tool_use_id": bloque.id, "content": resultado, "is_error": es_error}


def texto_de(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main() -> None:
    memoria = cargar_memoria()

    print("Chatbot v10 (v09 + prompt caching). Escribí 'salir' para terminar u 'olvidar' para borrar la memoria.")
    archivos = {frag["fuente"] for frag in INDICE}
    print(f"(Documentos indexados: {len(archivos)} archivos, {len(INDICE)} fragmentos)")
    if memoria:
        print(f"(Memoria cargada: {len(memoria)} mensajes de sesiones anteriores)")
    print()

    while True:
        try:
            user_input = input("Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        if user_input.lower() == "olvidar":
            memoria.clear()
            ARCHIVO_MEMORIA.unlink(missing_ok=True)
            print("\n(Memoria borrada, también del disco)\n")
            continue
        if not user_input:
            continue

        # Robustez (de la v08): cada error se explica y el chat sigue
        try:
            empezar_registro_de_turno()  # de la v09
            turno(memoria, user_input)
        except RespuestaRechazada:
            print("\n(Claude prefirió no responder a eso. Probá reformular la pregunta.)\n")
            continue
        except anthropic.RateLimitError:
            print("\n(La API está saturada: llegaste al límite de uso. Esperá un momento y repetí la pregunta.)\n")
            continue
        except anthropic.APIConnectionError:
            print("\n(No hay conexión con la API. Revisá tu internet y repetí la pregunta.)\n")
            continue
        except anthropic.APIStatusError as e:
            print(f"\n(Error de la API: {e.status_code} {e.message}. El turno se descartó.)\n")
            continue
        except KeyboardInterrupt:
            print("\n(Turno cancelado)\n")
            continue
        except Exception as e:  # cualquier otro problema, por ejemplo que falte la clave de la API
            print(f"\n(Error inesperado: {e.__class__.__name__}: {e}. El turno se descartó.)\n")
            continue

        print(resumen_del_turno())  # de la v09: qué costó este turno
        guardar_memoria(memoria)
        print()


if __name__ == "__main__":
    main()
