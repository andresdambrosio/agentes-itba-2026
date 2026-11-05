"""Versión 23: v22 + tareas largas, en varias sesiones, con archivo de progreso.

Qué agrega respecto de la v22 y por qué:
Hay trabajos que no entran en un turno ni en una conversación: armar una guía
completa para clientes nuevos, revisar todo el manual sección por sección.
Si el agente intenta hacerlos de una vez, el contexto se llena, la
compactación (v11) empieza a perder detalles y, si algo se corta, se pierde
todo. Los harness para tareas largas lo resuelven así:

    1. Un agente INICIALIZADOR no hace la tarea: la divide en pasos chicos e
       independientes y los guarda en un archivo de progreso
       (progreso_tarea.json). Comando: tarea <objetivo>
    2. En cada sesión, un agente TRABAJADOR arranca con el contexto vacío,
       lee el progreso, hace UN solo paso y registra el resultado en el
       archivo con registrar_avance. Comando: seguir
    3. Cuando no quedan pasos, el programa arma el documento final con los
       resultados y pide permiso para guardarlo en salida/ (como en la v18).

La memoria de trabajo es el ARCHIVO, no la conversación. Por eso se puede
cerrar el programa entre paso y paso, o puede fallar una sesión: alcanza con
volver a escribir 'seguir'. Y cada trabajador empieza limpio, sin arrastrar
lo que leyeron los anteriores.

Diferencias con versiones anteriores:
    - El plan de la v20 vive dentro de una conversación; el progreso de una
      tarea larga vive en un archivo y sobrevive a varias conversaciones.
    - Las notas de la v16 guardan datos del usuario; el progreso guarda el
      estado de un trabajo.
    - Un subagente (v13) hace una parte y vuelve en el mismo turno; un
      trabajador hace una parte por sesión y deja todo anotado para el próximo.

Comandos nuevos: 'tarea <objetivo>', 'tarea' (ver el progreso), 'seguir' y
'olvidar tarea'.

Uso:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v23.py
"""

import ast
import asyncio
import email.message
import hashlib
import itertools
import json
import math
import os
import operator
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

# max_retries (de la v08): el SDK reintenta solo los errores de red, 429 y 5xx
client = anthropic.Anthropic(max_retries=4)
MODEL = "claude-opus-5-5"  # el modelo por defecto (y el que hace los resúmenes)
MAX_TOKENS = 16000
CARPETA = Path(__file__).parent
ARCHIVO_MEMORIA = CARPETA / "memoria_v23.json"
ARCHIVO_PROGRESO = CARPETA / "progreso_tarea.json"  # v23: estado de la tarea larga en curso
ARCHIVO_REGISTRO = CARPETA / "registro.jsonl"     # v09: una línea por llamada y por herramienta
ARCHIVO_NOTAS = CARPETA / "notas.json"
ARCHIVO_EMBEDDINGS = CARPETA / "embeddings_cache.json"
CARPETA_SALIDA = CARPETA / "salida"                # archivos que escribe el bot
CARPETA_BANDEJA = CARPETA / "bandeja_salida"       # mails "enviados" (simulados)
ARCHIVO_MCP = CARPETA / "mcp_servidores.json"      # qué servidores MCP usar
ARCHIVO_LOG_MCP = CARPETA / "mcp_servidores.log"   # mensajes de los servidores

# Embeddings (de la v17)
MODELO_EMBEDDINGS = "voyage-3.5"

# Router (de la v12)
MODELO_CLASIFICADOR = "claude-haiku-5-5"
MODELOS = {
    "trivial": "claude-haiku-5-5",
    "normal": "claude-sonnet-5-5",
    "dificil": "claude-opus-5-5",
}

# Modelos que el agente principal puede elegir para un subagente (de la v13)
MODELOS_SUBAGENTE = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
CARPETA_DOCUMENTOS = CARPETA / "documentos"
CARPETA_SKILLS = CARPETA / "skills"

# Manejo del contexto (de la v11). Valores bajos a propósito para que
# la compactación se vea en una charla de prueba; en un programa real
# podrían ser mucho más altos.
LIMITE_CONTEXTO = 30_000   # tokens: si la conversación pasa de aquí, se compacta
TURNOS_A_CONSERVAR = 2     # turnos recientes que no se resumen
MAX_VUELTAS = 10           # tope de llamadas seguidas a herramientas por turno (v08)

SYSTEM_BASE = (
    "Sos Tecla, la especialista en reservas y viajes de Nimbo Viajes, amable y directa. "
    "Respondés siempre en español "
    "rioplatense (de vos), de forma breve y clara, sin rodeos. "
    "Para cualquier cuenta usá la herramienta calcular; nunca calcules de memoria. "
    "Para la fecha o la hora usá la herramienta obtener_hora. "
    "Tenés acceso a los documentos del usuario con la herramienta buscar_documentos. "
    "Si la pregunta puede estar respondida en esos documentos, buscala antes de "
    "responder y citá el archivo de donde sacaste la información. Si no encontrás "
    "nada relevante, decilo en lugar de inventar. "
    "Si una tarea requiere leer mucho o tiene partes independientes (por ejemplo, "
    "revisar varios temas del manual y resumirlos), usá la herramienta delegar para "
    "pasársela a un subagente: elegí haiku para tareas mecánicas, sonnet para las "
    "normales y opus solo para las que exigen razonar mucho. "
    "Si la consulta es sobre pagos, reembolsos ya pedidos, facturas o puntos Nube, "
    "derivala a facturacion con la herramienta derivar."
)

# Los otros agentes del equipo (de la v15)
SYSTEM_TRIAGE = (
    "Sos la recepción de Nimbo Viajes. Tu único trabajo es entender qué necesita el "
    "cliente y derivarlo con la herramienta derivar:\n"
    "- reservas: viajes, destinos, paquetes, equipaje, cambios y cancelaciones de "
    "viajes, presupuestos, reclamos y cualquier otra consulta sobre la empresa (horarios, "
    "oficinas, contactos, emergencias).\n"
    "- facturacion: pagos, facturas, reembolsos ya solicitados y puntos Nube.\n"
    "Si el mensaje es solo un saludo, saludá en una frase y preguntá en qué podés ayudar. "
    "Si no está claro qué necesita, hacé una sola pregunta. No respondas vos las consultas: "
    "derivalas. Respondés en español rioplatense (de vos)."
)

SYSTEM_FACTURACION = (
    "Sos el agente de facturación de Nimbo Viajes: pagos, facturas, reembolsos y el "
    "programa de puntos Nube. Respondés en español rioplatense (de vos), breve y claro. Si el cliente necesita "
    "que lo atienda una persona, buscale en los documentos el contacto que corresponde. "
    "Buscá las reglas en los documentos con buscar_documentos y citá la fuente; hacé las "
    "cuentas con calcular. Si la consulta es sobre viajes, destinos, equipaje o "
    "reclamos, derivala a reservas con la herramienta derivar."
)

SYSTEM_SUBAGENTE = (
    "Sos un subagente: otro asistente te encargó una tarea concreta. Resolvela usando "
    "tus herramientas y respondé en español con un resultado completo pero compacto, "
    "listo para que el otro asistente lo use. Incluí las fuentes de los datos que "
    "saques de documentos. No hagas preguntas: si falta información, decilo en el resultado."
)


# ---------------------------------------------------------------------------
# Skills (de la v14)
# ---------------------------------------------------------------------------

def leer_skill(archivo: Path) -> dict | None:
    """Lee un .md de skills/ y separa el encabezado (nombre, descripcion)
    del cuerpo con las instrucciones."""
    texto = archivo.read_text(encoding="utf-8")
    partes = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", texto, re.DOTALL)
    if not partes:
        return None
    encabezado = dict(
        linea.split(":", 1) for linea in partes.group(1).splitlines() if ":" in linea
    )
    encabezado = {k.strip(): v.strip() for k, v in encabezado.items()}
    return {
        "nombre": encabezado.get("nombre", archivo.stem),
        "descripcion": encabezado.get("descripcion", ""),
        "texto": partes.group(2).strip(),
    }


def indexar_skills() -> dict:
    skills = {}
    if CARPETA_SKILLS.exists():
        for archivo in sorted(CARPETA_SKILLS.glob("*.md")):
            skill = leer_skill(archivo)
            if skill:
                skills[skill["nombre"]] = skill
    return skills


SKILLS = indexar_skills()


def cargar_skill(nombre: str) -> str:
    """Devuelve las instrucciones completas de una skill."""
    skill = SKILLS.get(nombre)
    if not skill:
        raise ValueError(f"No existe la skill '{nombre}'. Disponibles: {', '.join(SKILLS) or 'ninguna'}")
    return skill["texto"]


def armar_system() -> str:
    """System prompt = instrucciones fijas + la LISTA de skills (sin su texto).
    Se arma una sola vez al arrancar, así no rompe la caché de la v10."""
    if not SKILLS:
        return SYSTEM_BASE
    lista = "\n".join(f"- {s['nombre']}: {s['descripcion']}" for s in SKILLS.values())
    return (
        f"{SYSTEM_BASE}\n\n"
        "Tenés skills: instrucciones detalladas para ciertos tipos de tarea. Cuando la "
        "tarea del usuario coincida con una, llamá a cargar_skill con su nombre ANTES de "
        "empezar y seguí esas instrucciones. Skills disponibles:\n"
        f"{lista}"
    )


SYSTEM = armar_system()


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


def puntuar_por_palabras(consulta: str) -> list[tuple[float, dict]]:
    """Búsqueda de la v07 (TF-IDF): puntaje de cada fragmento según las
    palabras que comparte con la consulta."""
    palabras_consulta = set(tokenizar(consulta))
    puntajes = []
    for frag in INDICE:
        puntaje = sum(
            IDF.get(p, 0) * frag["palabras"][p] / (frag["palabras"][p] + 1)
            for p in palabras_consulta
        )
        if puntaje > 0:
            puntajes.append((puntaje, frag))
    return puntajes


# ---------------------------------------------------------------------------
# Embeddings (de la v17)
# ---------------------------------------------------------------------------

def crear_cliente_embeddings():
    """Devuelve un cliente de Voyage AI, o None si no está configurado."""
    if not os.environ.get("VOYAGE_API_KEY"):
        return None
    try:
        import voyageai  # opcional: solo hace falta para esta búsqueda
    except ImportError:
        print("(Para buscar por significado instalá voyageai: pip install voyageai)")
        return None
    return voyageai.Client()  # lee VOYAGE_API_KEY del entorno


def clave_cache(texto: str) -> str:
    return hashlib.sha256(f"{MODELO_EMBEDDINGS}\n{texto}".encode()).hexdigest()


def indexar_embeddings(cliente) -> None:
    """Calcula el embedding de cada fragmento y lo guarda en frag["vector"].
    Los ya calculados se leen de embeddings_cache.json."""
    cache = {}
    if ARCHIVO_EMBEDDINGS.exists():
        try:
            cache = json.loads(ARCHIVO_EMBEDDINGS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    faltan = [f for f in INDICE if clave_cache(f["texto"]) not in cache]
    for i in range(0, len(faltan), 64):  # de a tandas, para no mandar todo junto
        tanda = faltan[i:i + 64]
        resultado = cliente.embed([f["texto"] for f in tanda],
                                  model=MODELO_EMBEDDINGS, input_type="document")
        for frag, vector in zip(tanda, resultado.embeddings):
            cache[clave_cache(frag["texto"])] = vector
    if faltan:
        ARCHIVO_EMBEDDINGS.write_text(json.dumps(cache), encoding="utf-8")
    for frag in INDICE:
        frag["vector"] = cache[clave_cache(frag["texto"])]


def similitud_coseno(a: list[float], b: list[float]) -> float:
    """1 = mismo significado, 0 = nada que ver."""
    producto = sum(x * y for x, y in zip(a, b))
    normas = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return producto / normas if normas else 0.0


def puntuar_por_significado(consulta: str) -> list[tuple[float, dict]]:
    vector = CLIENTE_EMBEDDINGS.embed([consulta], model=MODELO_EMBEDDINGS,
                                      input_type="query").embeddings[0]
    return [(similitud_coseno(vector, frag["vector"]), frag) for frag in INDICE]


def preparar_embeddings():
    """Intenta activar la búsqueda por significado. Si no se puede, devuelve
    None y se usa la búsqueda por palabras."""
    cliente = crear_cliente_embeddings()
    if cliente is None or not INDICE:
        return None
    try:
        indexar_embeddings(cliente)
        return cliente
    except Exception as e:  # sin red, clave inválida, etc.
        print(f"(No pude calcular los embeddings: {e}. Uso búsqueda por palabras.)")
        return None


CLIENTE_EMBEDDINGS = preparar_embeddings()
MODO_BUSQUEDA = "significado" if CLIENTE_EMBEDDINGS else "palabras"


def buscar_documentos(consulta: str, cantidad: int = 3, modo: str | None = None) -> str:
    """Devuelve los fragmentos más relevantes para la consulta. Usa embeddings
    si están disponibles; si no, palabras clave."""
    if not INDICE:
        return "No hay documentos cargados en la carpeta documentos/."
    modo = modo or MODO_BUSQUEDA
    puntajes = puntuar_por_significado(consulta) if modo == "significado" else puntuar_por_palabras(consulta)

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


# Herramientas básicas: las tienen el principal y los subagentes
HERRAMIENTAS_BASE = [
    {
        "name": "buscar_documentos",
        "description": "Busca información en los documentos del usuario y devuelve los "
                       "fragmentos más relevantes con su fuente. "
                       + ("La búsqueda es por significado: describí con tus palabras lo que "
                          "necesitás saber. " if MODO_BUSQUEDA == "significado" else
                          "La búsqueda es por palabras clave: usá términos concretos que "
                          "probablemente aparezcan en el texto. ")
                       + "Podés llamarla varias veces con distintas consultas si no encontrás algo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Lo que buscás, por ejemplo 'cuánto devuelven si cancelo el viaje'",
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

# La herramienta delegar (de la v13): solo la tiene el principal,
# así un subagente no puede lanzar subagentes sin fin.
HERRAMIENTA_DELEGAR = {
    "name": "delegar",
    "description": "Encarga una tarea a un subagente: otro asistente con las mismas herramientas "
                   "básicas (buscar_documentos, calcular, obtener_hora) pero con la memoria vacía. "
                   "No ve esta conversación, así que la tarea debe explicarse sola, con todos los "
                   "datos necesarios. Devuelve solo su resultado final. Usala para tareas que "
                   "requieren mucha lectura o búsquedas, así no se llena esta conversación.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tarea": {
                "type": "string",
                "description": "Qué tiene que hacer el subagente y qué debe devolver.",
            },
            "modelo": {
                "type": "string",
                "enum": list(MODELOS_SUBAGENTE),
                "description": "haiku: tareas simples y mecánicas. sonnet: tareas normales. "
                               "opus: tareas que exigen mucho razonamiento.",
            },
        },
        "required": ["tarea", "modelo"],
    },
}

# La herramienta cargar_skill (de la v14)
HERRAMIENTA_CARGAR_SKILL = {
    "name": "cargar_skill",
    "description": "Devuelve las instrucciones completas de una skill de la lista del "
                   "system prompt. Usala antes de empezar una tarea que coincida con una skill.",
    "input_schema": {
        "type": "object",
        "properties": {
            "nombre": {"type": "string", "description": "Nombre exacto de la skill, por ejemplo 'redactar_reclamo'"},
        },
        "required": ["nombre"],
    },
}

# ---------------------------------------------------------------------------
# Memoria a largo plazo: notas (de la v16)
# ---------------------------------------------------------------------------

def cargar_notas() -> list[str]:
    if not ARCHIVO_NOTAS.exists():
        return []
    try:
        return json.loads(ARCHIVO_NOTAS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


NOTAS = cargar_notas()  # las de esta sesión; las nuevas se agregan al final


def guardar_nota(nota: str) -> str:
    nota = nota.strip()
    if not nota:
        raise ValueError("La nota está vacía")
    if nota in NOTAS:
        return "Esa nota ya estaba guardada."
    NOTAS.append(nota)
    ARCHIVO_NOTAS.write_text(json.dumps(NOTAS, ensure_ascii=False, indent=2), encoding="utf-8")
    return "Nota guardada. La vas a recordar en las próximas sesiones."


def con_notas(system: str) -> str:
    """Agrega las notas al final de un system prompt. Se llama una sola vez,
    al arrancar, con las notas que había en ese momento."""
    texto = (
        f"{system}\n\n"
        "Memoria a largo plazo: cuando el usuario cuente algo de sí mismo que convenga "
        "recordar en futuras conversaciones (su nombre, preferencias, datos de sus viajes), "
        "guardalo con guardar_nota en una frase corta. No guardes datos sensibles como "
        "contraseñas o números de tarjeta."
    )
    if NOTAS:
        texto += "\n\nLo que sabés del usuario por conversaciones anteriores:\n"
        texto += "\n".join(f"- {n}" for n in NOTAS)
    return texto


HERRAMIENTA_GUARDAR_NOTA = {
    "name": "guardar_nota",
    "description": "Guarda un dato sobre el usuario para recordarlo en futuras conversaciones, "
                   "por ejemplo 'Se llama Andrés' o 'Prefiere viajar en temporada baja'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "nota": {"type": "string", "description": "El dato, en una frase corta y en tercera persona."},
        },
        "required": ["nota"],
    },
}


# ---------------------------------------------------------------------------
# Herramientas con efectos, que piden confirmación (de la v18)
# ---------------------------------------------------------------------------

SYSTEM_ACCIONES = (
    " Podés guardar archivos para el usuario con escribir_archivo y enviar mails con "
    "enviar_mail. El programa le pedirá permiso al usuario antes de ejecutarlas; si no lo "
    "autoriza, no insistas: preguntale qué quiere cambiar."
)


def escribir_archivo(nombre: str, contenido: str) -> str:
    """Guarda un archivo de texto en salida/. Solo acepta un nombre simple
    (sin carpetas) para que el bot no pueda escribir en otro lugar."""
    if Path(nombre).name != nombre or nombre.startswith(".") or Path(nombre).suffix not in (".txt", ".md"):
        raise ValueError("Usá un nombre simple que termine en .txt o .md, sin carpetas")
    CARPETA_SALIDA.mkdir(exist_ok=True)
    destino = CARPETA_SALIDA / nombre
    destino.write_text(contenido, encoding="utf-8")
    return f"Archivo guardado en {destino.relative_to(CARPETA)}"


def enviar_mail(para: str, asunto: str, cuerpo: str) -> str:
    """Simula el envío: guarda el mail como .eml en bandeja_salida/."""
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", para):
        raise ValueError(f"Dirección de correo inválida: {para}")
    mensaje = email.message.EmailMessage()
    mensaje["To"], mensaje["From"], mensaje["Subject"] = para, "tecla@ejemplo.local", asunto
    mensaje.set_content(cuerpo)
    CARPETA_BANDEJA.mkdir(exist_ok=True)
    destino = CARPETA_BANDEJA / f"{datetime.now():%Y%m%d-%H%M%S-%f}.eml"
    destino.write_bytes(bytes(mensaje))
    # Para enviarlo de verdad: smtplib.SMTP_SSL(servidor).send_message(mensaje)
    return f"Mail enviado a {para} (simulado: guardado en {destino.relative_to(CARPETA)})"


HERRAMIENTAS_ACCIONES = [
    {
        "name": "escribir_archivo",
        "description": "Guarda un archivo de texto (.txt o .md) para el usuario en la carpeta "
                       "salida/. Requiere la autorización del usuario.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del archivo, por ejemplo 'presupuesto_lisboa.md'"},
                "contenido": {"type": "string", "description": "Texto completo del archivo"},
            },
            "required": ["nombre", "contenido"],
        },
    },
    {
        "name": "enviar_mail",
        "description": "Envía un correo electrónico. Requiere la autorización del usuario.",
        "input_schema": {
            "type": "object",
            "properties": {
                "para": {"type": "string", "description": "Dirección de destino"},
                "asunto": {"type": "string"},
                "cuerpo": {"type": "string", "description": "Texto del mail"},
            },
            "required": ["para", "asunto", "cuerpo"],
        },
    },
]
REQUIEREN_CONFIRMACION = {h["name"] for h in HERRAMIENTAS_ACCIONES}


def necesita_confirmacion(nombre: str) -> bool:
    """Qué herramientas piden permiso antes de ejecutarse. De la v19:
    además de las acciones de la v18, todas las herramientas MCP, salvo las de
    solo lectura de servidores marcados como confiables. Sin esto, activar un
    servidor que escribe archivos saltearía la confirmación de la v18."""
    return nombre in REQUIEREN_CONFIRMACION or MCP.pide_confirmacion(nombre)


def pedir_confirmacion(nombre: str, argumentos: dict) -> bool:
    """El humano en el bucle: muestra la acción y espera un sí o un no.
    Si el programa no corre en modo interactivo (por ejemplo, en evaluar.py),
    la respuesta es siempre no."""
    if not estado["interactivo"]:
        return False
    print(f"\n  [confirmar] El bot quiere usar {nombre}:")
    # Mostramos TODO el contenido, sin recortar: la persona tiene que poder leer
    # exactamente lo que autoriza (un mail recortado no se puede revisar).
    for clave, valor in argumentos.items():
        lineas = str(valor).splitlines() or [""]
        print(f"      {clave}: {lineas[0]}")
        for linea in lineas[1:]:
            print(f"      {' ' * len(clave)}  {linea}")
    try:
        respuesta = input("  ¿Lo autorizás? (s/n): ").strip().lower()
    except EOFError:
        return False
    return respuesta in ("s", "si", "sí", "y", "yes")


# ---------------------------------------------------------------------------
# MCP: herramientas de servidores externos (de la v19)
# ---------------------------------------------------------------------------

class ConexionesMCP:
    """Mantiene abiertas las conexiones con los servidores MCP.

    La librería mcp es asíncrona, así que corremos un event loop de asyncio
    en un hilo aparte. Cada servidor tiene una tarea que abre la conexión y
    la mantiene abierta hasta que se llama a cerrar()."""

    def __init__(self):
        self.herramientas = []   # definiciones para Claude
        self.destinos = {}       # nombre para Claude -> (servidor, nombre en el servidor)
        self.sesiones = {}       # servidor -> sesión abierta
        self.solo_lectura = set()  # herramientas que no piden confirmación (ver pide_confirmacion)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._cerrar = None
        self._tareas = []

    def _esperar(self, corrutina, segundos: float = 60):
        """Corre una corrutina en el hilo de MCP y espera el resultado."""
        return asyncio.run_coroutine_threadsafe(corrutina, self.loop).result(segundos)

    def conectar(self, config: dict) -> None:
        """Lanza y conecta los servidores activos de la configuración."""
        try:
            from mcp import StdioServerParameters  # opcional: solo hace falta en la v19
        except ImportError:
            print("(Para usar servidores MCP instalá la librería: pip install mcp)")
            return
        self._esperar(self._crear_evento())
        for servidor, datos in config.items():
            if not datos.get("activo", True):
                continue
            comando = sys.executable if datos["command"] == "python" else datos["command"]
            parametros = StdioServerParameters(command=comando, args=datos.get("args", []), cwd=CARPETA)
            confiable = bool(datos.get("confiable", False))
            try:
                self._esperar(self._conectar_uno(servidor, parametros, confiable))
            except Exception as e:
                print(f"(No pude conectar con el servidor MCP '{servidor}': {e})")

    async def _crear_evento(self):
        self._cerrar = asyncio.Event()

    async def _conectar_uno(self, servidor: str, parametros, confiable: bool) -> None:
        lista = asyncio.get_running_loop().create_future()
        self._tareas.append(asyncio.create_task(self._mantener(servidor, parametros, lista)))
        sesion = await lista  # espera a que la conexión esté lista (o falle)
        self.sesiones[servidor] = sesion
        for herramienta in (await sesion.list_tools()).tools:
            nombre = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{servidor}__{herramienta.name}")[:64]
            self.destinos[nombre] = (servidor, herramienta.name)
            # Cada herramienta MCP puede declarar "anotaciones", por ejemplo
            # read_only_hint=True si no modifica nada. Son PISTAS que declara
            # el propio servidor: solo les creemos si el servidor está marcado
            # como confiable en mcp_servidores.json.
            anotaciones = getattr(herramienta, "annotations", None)
            if confiable and anotaciones and getattr(anotaciones, "read_only_hint", False):
                self.solo_lectura.add(nombre)
            self.herramientas.append({
                "name": nombre,
                "description": f"[Servidor MCP '{servidor}'] {herramienta.description or ''}".strip(),
                "input_schema": herramienta.input_schema,
            })

    async def _mantener(self, servidor: str, parametros, lista: asyncio.Future) -> None:
        """Abre la conexión y la deja abierta hasta que se pida cerrarla.
        Abrir y cerrar tienen que pasar en la misma tarea, por eso es una
        tarea que vive todo el programa."""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        try:
            with open(ARCHIVO_LOG_MCP, "a", encoding="utf-8") as log:
                async with stdio_client(parametros, errlog=log) as (lectura, escritura):
                    async with ClientSession(lectura, escritura) as sesion:
                        await sesion.initialize()
                        lista.set_result(sesion)
                        await self._cerrar.wait()
        except Exception as e:
            if not lista.done():
                lista.set_exception(e)

    def pide_confirmacion(self, nombre: str) -> bool:
        """True si es una herramienta MCP que puede cambiar algo (o de la que
        no sabemos si cambia algo): esas pasan por el "sí" de la v18."""
        return nombre in self.destinos and nombre not in self.solo_lectura

    def llamar(self, nombre: str, argumentos: dict) -> str:
        """Le pide al servidor que ejecute una herramienta y devuelve el texto."""
        servidor, original = self.destinos[nombre]
        resultado = self._esperar(self.sesiones[servidor].call_tool(original, argumentos))
        partes = [c.text if getattr(c, "type", "") == "text" else f"[contenido {getattr(c, 'type', '?')}]"
                  for c in getattr(resultado, "content", [])]
        texto = "\n".join(partes) or "(sin resultado)"
        if getattr(resultado, "is_error", False):
            raise RuntimeError(texto)
        return texto

    def cerrar(self) -> None:
        """Cierra las conexiones (y con ellas, los procesos de los servidores)."""
        if self._cerrar is None:
            return

        async def terminar():
            self._cerrar.set()
            await asyncio.gather(*self._tareas, return_exceptions=True)

        try:
            self._esperar(terminar(), segundos=10)
        except Exception:
            pass


def cargar_config_mcp() -> dict:
    if not ARCHIVO_MCP.exists():
        return {}
    try:
        return json.loads(ARCHIVO_MCP.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"(mcp_servidores.json tiene un error: {e})")
        return {}


MCP = ConexionesMCP()
MCP.conectar(cargar_config_mcp())


# ---------------------------------------------------------------------------
# Herramientas en paralelo (de la v22)
# ---------------------------------------------------------------------------

MAX_EN_PARALELO = 5
# Cambian el estado del programa: si corrieran a la vez podrían pisarse
CAMBIAN_EL_ESTADO = {"derivar", "guardar_nota", "actualizar_plan"}

SYSTEM_PARALELO = (
    " Si una tarea tiene partes independientes (por ejemplo, averiguar lo mismo para "
    "varios destinos), pedí todas las herramientas juntas en la misma respuesta, por "
    "ejemplo un delegar por destino: el programa las ejecuta a la vez."
)


def puede_ir_en_paralelo(nombre: str) -> bool:
    """Las que piden confirmación van de a una (no se pueden hacer dos
    preguntas a la vez por pantalla), igual que las que cambian el estado."""
    return not necesita_confirmacion(nombre) and nombre not in CAMBIAN_EL_ESTADO


def ejecutar_medido(bloque, prefijo: str) -> tuple[dict, float]:
    inicio = time.monotonic()
    return resultado_herramienta(bloque, prefijo), time.monotonic() - inicio


def ejecutar_pedidos(pedidos: list, prefijo: str) -> list[dict]:
    """Ejecuta las herramientas que Claude pidió en una misma respuesta: las
    que pueden ir a la vez, en paralelo; las demás, después y de a una.
    Devuelve los resultados en el mismo orden que los pedidos (la API los
    empareja por tool_use_id, pero el orden hace más fácil leer el historial)."""
    resultados = [None] * len(pedidos)
    juntos = [i for i, b in enumerate(pedidos) if puede_ir_en_paralelo(b.name)]
    if len(juntos) > 1:
        inicio = time.monotonic()
        # Al salir del "with" se espera a que terminen todos los hilos. Por eso
        # un Ctrl+C durante esta espera no corta los que ya arrancaron.
        with ThreadPoolExecutor(max_workers=MAX_EN_PARALELO) as grupo:
            futuros = {i: grupo.submit(ejecutar_medido, pedidos[i], prefijo) for i in juntos}
        duraciones = []
        for i, futuro in futuros.items():
            resultados[i], segundos = futuro.result()
            duraciones.append(segundos)
        if estado["mostrar"]:
            print(f"{prefijo}[paralelo] {len(juntos)} herramientas a la vez: "
                  f"{time.monotonic() - inicio:.1f} s (una tras otra, ~{sum(duraciones):.1f} s)")
    for i, bloque in enumerate(pedidos):
        if resultados[i] is None:
            resultados[i] = resultado_herramienta(bloque, prefijo)
    return resultados


# ---------------------------------------------------------------------------
# Planificación (de la v20)
# ---------------------------------------------------------------------------

ESTADOS_PASO = {"pendiente": "[ ]", "en_curso": "[>]", "hecho": "[x]"}

SYSTEM_PLAN = (
    " Para tareas de varios pasos (un presupuesto, comparar destinos, un reclamo, "
    "una consulta con varias preguntas), antes de empezar armá un plan con "
    "actualizar_plan y actualizalo a medida que avanzás: marcá en_curso el paso que "
    "estás haciendo y hecho cuando lo terminás. Para preguntas simples no hace falta plan."
)

HERRAMIENTA_PLAN = {
    "name": "actualizar_plan",
    "description": "Crea o actualiza el plan de la tarea en curso: la lista de pasos, cada uno "
                   "con su estado. Mandá siempre la lista completa, no solo lo que cambió. "
                   "Usala para tareas de varios pasos, no para preguntas simples.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pasos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paso": {"type": "string", "description": "Qué hay que hacer, en pocas palabras"},
                        "estado": {"type": "string", "enum": list(ESTADOS_PASO)},
                    },
                    "required": ["paso", "estado"],
                },
            }
        },
        "required": ["pasos"],
    },
}


def plan_como_texto(plan: list) -> str:
    return "\n".join(f"{ESTADOS_PASO[p['estado']]} {p['paso']}" for p in plan)


def actualizar_plan(pasos: list) -> str:
    """Reemplaza el plan entero. Que el modelo mande siempre la lista completa
    evita tener que llevar la cuenta de qué cambió. El programa valida que el
    plan tenga sentido: si no, Claude recibe el error y lo corrige."""
    if not pasos:
        raise ValueError("El plan está vacío")
    for p in pasos:
        if not str(p.get("paso", "")).strip():
            raise ValueError("Hay un paso sin descripción")
        if p.get("estado") not in ESTADOS_PASO:
            raise ValueError(f"Estado inválido: {p.get('estado')}. Usá: {', '.join(ESTADOS_PASO)}")
    if sum(p["estado"] == "en_curso" for p in pasos) > 1:
        raise ValueError("Solo un paso puede estar en_curso a la vez: terminá uno antes de empezar otro")
    estado["plan"] = [{"paso": str(p["paso"]).strip(), "estado": p["estado"]} for p in pasos]
    if estado["mostrar"]:
        print("  [plan]\n" + "\n".join("      " + linea for linea in plan_como_texto(estado["plan"]).splitlines()))
    hechos = sum(p["estado"] == "hecho" for p in estado["plan"])
    return f"Plan actualizado ({hechos}/{len(estado['plan'])} pasos hechos):\n{plan_como_texto(estado['plan'])}"


def avisar_pasos_pendientes() -> None:
    """Al terminar el turno, el programa revisa el plan: si el agente respondió
    pero quedaron pasos sin hacer, lo avisa. Un harness no confía en que el
    modelo se acuerde de todo; lo comprueba."""
    pendientes = [p["paso"] for p in estado["plan"] if p["estado"] != "hecho"]
    if pendientes and estado["mostrar"]:
        print(f"  [plan] Pasos sin terminar ({len(pendientes)}): {'; '.join(pendientes)}")


# ---------------------------------------------------------------------------
# Verificación antes de responder (de la v21)
# ---------------------------------------------------------------------------

MODELO_VERIFICADOR = "claude-sonnet-5-5"
MAX_REVISIONES = 1  # con 1, la respuesta corregida ya no se vuelve a revisar
HERRAMIENTAS_CON_EVIDENCIA = {"buscar_documentos", "delegar"}

SYSTEM_VERIFICADOR = (
    "Revisás respuestas de un asistente de Nimbo Viajes antes de que lleguen al cliente. "
    "Compará la respuesta con lo que devolvieron las herramientas que el asistente usó "
    "(fragmentos del manual, la fecha, cuentas, conversiones). Marcala como incorrecta solo "
    "si contradice esos resultados, afirma un dato de Nimbo Viajes que no está en ellos o "
    "tiene una cuenta mal hecha. Ofrecer algo que figura en 'Lo que el "
    "asistente puede hacer' no es inventar un dato. No juzgues el estilo ni el largo. Si "
    "está bien, dejá 'problemas' vacío; si no, explicá el problema en una o dos frases."
)


def evidencia_del_turno(mensajes: list) -> list[str]:
    """Junta lo que devolvieron las herramientas en este turno: contra eso se
    compara la respuesta. Solo hay revisión si el turno usó documentos
    (búsquedas o subagentes), pero entonces van TODOS los resultados, cada uno
    con el nombre de su herramienta: de la v21: si el revisor solo viera los
    fragmentos, marcaría como inventados la fecha de obtener_hora, una cuenta
    de calcular o una conversión de moneda de un servidor MCP."""
    nombres = {b["id"]: b["name"] for m in mensajes
               if m["role"] == "assistant" and isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_use"}
    if not any(n in HERRAMIENTAS_CON_EVIDENCIA for n in nombres.values()):
        return []
    return [f"[{nombres[b['tool_use_id']]}] {b.get('content', '')}" for m in mensajes
            if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result" and b.get("tool_use_id") in nombres]


def capacidades_del_agente() -> str:
    """De la v21: lo que el agente activo puede ofrecer: sus herramientas y las
    skills. Sin esto, el revisor solo ve los fragmentos del manual y marca
    como inventado que el bot ofrezca "armarte un presupuesto" (una skill)."""
    def primera_frase(texto: str) -> str:
        return " ".join(texto.split(". ")[0].split())
    lineas = [f"- {h['name']}: {primera_frase(h.get('description', ''))}"
              for h in AGENTES[estado["agente"]]["herramientas"]]
    lineas += [f"- skill {s['nombre']}: {s['descripcion']}" for s in SKILLS.values()]
    return "\n".join(lineas)


def revisar(pregunta: str, respuesta: str, evidencia: list[str]) -> tuple[bool, str]:
    """El revisor: otro modelo, con otro system prompt, que solo ve la
    pregunta, los fragmentos, lo que el asistente puede hacer y la respuesta."""
    inicio = time.monotonic()
    response = client.messages.create(
        model=MODELO_VERIFICADOR,
        max_tokens=2000,
        output_config={
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"correcta": {"type": "boolean"}, "problemas": {"type": "string"}},
                    "required": ["correcta", "problemas"],
                    "additionalProperties": False,
                },
            },
        },
        system=SYSTEM_VERIFICADOR,
        messages=[{"role": "user", "content":
                   f"Pregunta del cliente:\n{pregunta}\n\n"
                   f"Lo que devolvieron las herramientas:\n{chr(10).join(evidencia)[:8000]}\n\n"
                   f"Lo que el asistente puede hacer:\n{capacidades_del_agente()}\n\n"
                   f"Respuesta a revisar:\n{respuesta}"}],
    )
    registrar_llamada(response, inicio)  # registro de la v09
    datos = json.loads("".join(b.text for b in response.content if b.type == "text"))
    return bool(datos["correcta"]), str(datos.get("problemas", ""))


def verificar_y_corregir(memoria: list, inicio: int, pregunta: str, respuesta: str,
                         modelo: str, mostrar: bool) -> str:
    """Revisa la respuesta del turno y, si tiene un problema, le pide al agente
    que la corrija. Devuelve la respuesta final."""
    for _ in range(MAX_REVISIONES):
        evidencia = evidencia_del_turno(memoria[inicio:])
        if not evidencia or not respuesta.strip():
            return respuesta  # nada contra qué comparar: no gastamos una llamada
        try:
            correcta, problemas = revisar(pregunta, respuesta, evidencia)
        except (anthropic.APIError, json.JSONDecodeError, KeyError):
            return respuesta  # si el revisor falla, entregamos la respuesta como está
        if correcta:
            if mostrar:
                print("  [verificación] La respuesta coincide con los documentos.")
            return respuesta
        if mostrar:
            print(f"  [verificación] Problema: {problemas}")
            print("  [verificación] Le pido al agente que corrija.")
        # La observación va como bloque de texto (no como texto suelto): así no
        # cuenta como un mensaje del cliente para la compactación (v11) ni para
        # el router (v12).
        memoria.append({"role": "user", "content": [{"type": "text", "text":
            f"[Revisión automática, no la escribió el cliente] Tu respuesta tiene este problema: "
            f"{problemas} Corregila usando los documentos y respondele de nuevo al cliente, "
            "sin mencionar la revisión."}]})
        respuesta = _bucle_agente(memoria, modelo, estado["agente"], mostrar)
    return respuesta


# La herramienta derivar (de la v15): la tienen todos los agentes del equipo
HERRAMIENTA_DERIVAR = {
    "name": "derivar",
    "description": "Pasa la conversación a otro agente del equipo. Desde ese momento el otro "
                   "agente atiende al usuario directamente. Usala cuando la consulta "
                   "corresponda a otro agente; no le avises al usuario antes, solo derivá.",
    "input_schema": {
        "type": "object",
        "properties": {
            "destino": {"type": "string", "enum": ["triage", "reservas", "facturacion"]},
            "motivo": {"type": "string", "description": "Qué necesita el usuario, en una frase, "
                                                       "para que el otro agente no tenga que preguntar."},
        },
        "required": ["destino", "motivo"],
    },
}

# El equipo: cada agente es solo un system prompt + una lista de herramientas
# Todos los agentes del equipo reciben las notas y pueden guardar notas nuevas
AGENTES = {
    "triage": {
        "system": con_notas(SYSTEM_TRIAGE),
        "herramientas": [HERRAMIENTA_DERIVAR, HERRAMIENTA_GUARDAR_NOTA],
    },
    # v20: reservas y facturación suman la planificación (triage solo deriva)
    "reservas": {
        "system": con_notas(SYSTEM + SYSTEM_ACCIONES + SYSTEM_PLAN + SYSTEM_PARALELO),  # v22: + paralelo
        "herramientas": HERRAMIENTAS_BASE + [HERRAMIENTA_DELEGAR, HERRAMIENTA_CARGAR_SKILL,
                                             HERRAMIENTA_DERIVAR, HERRAMIENTA_GUARDAR_NOTA]
                        + HERRAMIENTAS_ACCIONES + MCP.herramientas + [HERRAMIENTA_PLAN],
    },
    "facturacion": {
        "system": con_notas(SYSTEM_FACTURACION + SYSTEM_ACCIONES + SYSTEM_PLAN + SYSTEM_PARALELO),
        "herramientas": HERRAMIENTAS_BASE + [HERRAMIENTA_DERIVAR, HERRAMIENTA_GUARDAR_NOTA]
                        + HERRAMIENTAS_ACCIONES + MCP.herramientas + [HERRAMIENTA_PLAN],
    },
    # No es parte del equipo: es el que usa delegar (v13). No puede derivar.
    "subagente": {
        "system": SYSTEM_SUBAGENTE,
        "herramientas": HERRAMIENTAS_BASE,
    },
}
AGENTE_INICIAL = "triage"
EQUIPO = {"triage", "reservas", "facturacion"}  # v23: los que atienden al usuario y se derivan entre sí


# ---------------------------------------------------------------------------
# Tareas largas, en varias sesiones (lo nuevo de la v23)
# ---------------------------------------------------------------------------

MAX_PASOS_TAREA = 10
MODELO_INICIALIZADOR = MODELOS["dificil"]  # dividir bien la tarea es lo más difícil
MODELO_TRABAJADOR = MODELOS["normal"]      # cada paso es chico

SYSTEM_INICIALIZADOR = (
    "Sos el agente que prepara una tarea larga para Nimbo Viajes. No la hacés vos: la "
    "dividís en pasos chicos que otro agente va a hacer de a uno, en sesiones distintas "
    "y sin ver esta conversación. Cada paso tiene que poder hacerse por separado con las "
    "herramientas buscar_documentos y calcular, y producir una sección del resultado "
    f"final. Usá entre 2 y {MAX_PASOS_TAREA} pasos. Si te sirve, buscá en los documentos "
    "qué temas hay. Cuando tengas los pasos, llamá a crear_tarea una sola vez y "
    "respondé en una frase, en español rioplatense (de vos)."
)

SYSTEM_TRABAJADOR = (
    "Sos el agente que avanza una tarea larga de Nimbo Viajes, de a un paso por sesión. "
    "Empezás sin memoria: todo lo que sabés de la tarea está en el progreso que te pasan. "
    "Hacé SOLO el paso indicado, buscando en los documentos y haciendo las cuentas con "
    "calcular. Al terminar, llamá a registrar_avance con el resultado de ese paso, listo "
    "para pegar en el documento final: en Markdown, con un título de sección y citando la "
    "fuente. Después respondé en una frase, en español rioplatense (de vos)."
)

HERRAMIENTA_CREAR_TAREA = {
    "name": "crear_tarea",
    "description": "Guarda la tarea larga dividida en pasos, para que otros agentes la hagan "
                   "de a un paso por sesión.",
    "input_schema": {
        "type": "object",
        "properties": {
            "titulo": {"type": "string", "description": "Título del documento final"},
            "pasos": {"type": "array", "items": {"type": "string"},
                      "description": "Los pasos, en orden; cada uno se tiene que poder hacer por separado"},
        },
        "required": ["titulo", "pasos"],
    },
}

HERRAMIENTA_REGISTRAR_AVANCE = {
    "name": "registrar_avance",
    "description": "Guarda el resultado del paso que te tocó y lo marca como hecho. "
                   "Llamala una sola vez, al terminar el paso.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resultado": {"type": "string", "description": "El resultado del paso, en Markdown"},
        },
        "required": ["resultado"],
    },
}

# Dos agentes nuevos que no atienden al usuario (por eso no están en EQUIPO)
AGENTES["inicializador"] = {
    "system": SYSTEM_INICIALIZADOR,
    "herramientas": [h for h in HERRAMIENTAS_BASE if h["name"] == "buscar_documentos"]
                    + [HERRAMIENTA_CREAR_TAREA],
}
AGENTES["trabajador"] = {
    "system": SYSTEM_TRABAJADOR,
    "herramientas": HERRAMIENTAS_BASE + [HERRAMIENTA_REGISTRAR_AVANCE],
}


def leer_progreso() -> dict | None:
    if not ARCHIVO_PROGRESO.exists():
        return None
    try:
        return json.loads(ARCHIVO_PROGRESO.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def guardar_progreso(progreso: dict) -> None:
    ARCHIVO_PROGRESO.write_text(json.dumps(progreso, ensure_ascii=False, indent=2), encoding="utf-8")


def progreso_como_texto(progreso: dict) -> str:
    hechos = sum(p["estado"] == "hecho" for p in progreso["pasos"])
    lineas = [f"{progreso['titulo']} ({hechos}/{len(progreso['pasos'])} pasos, "
              f"{progreso['sesiones']} sesiones de trabajo)"]
    lineas += [f"  {ESTADOS_PASO[p['estado']]} {p['paso']}" for p in progreso["pasos"]]
    return "\n".join(lineas)


def crear_tarea(titulo: str, pasos: list) -> str:
    """La usa el inicializador: guarda la tarea en el archivo de progreso."""
    pasos = [str(p).strip() for p in pasos if str(p).strip()]
    if not 1 <= len(pasos) <= MAX_PASOS_TAREA:
        raise ValueError(f"La tarea tiene que tener entre 1 y {MAX_PASOS_TAREA} pasos")
    guardar_progreso({
        "titulo": titulo.strip() or "Tarea", "objetivo": estado["objetivo_tarea"],
        "creada": datetime.now().isoformat(timespec="seconds"), "sesiones": 0,
        "pasos": [{"paso": p, "estado": "pendiente", "resultado": ""} for p in pasos],
    })
    return f"Tarea creada con {len(pasos)} pasos."


def registrar_avance(resultado: str) -> str:
    """La usa el trabajador: guarda el resultado del paso que le tocó. El
    programa sabe cuál es (estado["paso_actual"]), así el modelo no puede
    marcar como hecho un paso que no hizo."""
    progreso, i = leer_progreso(), estado["paso_actual"]
    if progreso is None or i is None:
        raise ValueError("No hay ningún paso en curso")
    if progreso["pasos"][i]["estado"] == "hecho":
        raise ValueError("Este paso ya está registrado: respondé y terminá la sesión")
    if not resultado.strip():
        raise ValueError("El resultado está vacío")
    progreso["pasos"][i].update(estado="hecho", resultado=resultado.strip())
    guardar_progreso(progreso)  # se guarda YA: si después algo falla, el paso no se pierde
    return f"Paso {i + 1}/{len(progreso['pasos'])} registrado."


def empezar_tarea(objetivo: str) -> None:
    """Sesión del inicializador: divide la tarea y crea el archivo de progreso."""
    progreso = leer_progreso()
    if progreso and any(p["estado"] != "hecho" for p in progreso["pasos"]):
        print(f"\n(Ya hay una tarea en curso:\n{progreso_como_texto(progreso)}\n"
              "Escribí 'seguir' para avanzarla u 'olvidar tarea' para descartarla.)\n")
        return
    estado["objetivo_tarea"] = objetivo
    memoria_inicializador = [{"role": "user", "content": f"Tarea larga a preparar: {objetivo}"}]
    _bucle_agente(memoria_inicializador, MODELO_INICIALIZADOR, "inicializador",
                  mostrar=True, prefijo="  [inicializador] ")
    progreso = leer_progreso()
    if progreso and progreso.get("objetivo") == objetivo:
        print(f"\n  [tarea] Lista para empezar:\n{progreso_como_texto(progreso)}\n"
              "  [tarea] Escribí 'seguir' para hacer el primer paso.")
    else:
        print("\n  [tarea] El inicializador no creó la tarea. Probá describirla de otra forma.")


def seguir_tarea() -> None:
    """Una sesión de trabajo: un agente con contexto vacío hace UN paso."""
    progreso = leer_progreso()
    if progreso is None:
        print("\n(No hay ninguna tarea. Empezá una con: tarea <objetivo>)\n")
        return
    pendientes = [i for i, p in enumerate(progreso["pasos"]) if p["estado"] != "hecho"]
    if not pendientes:
        entregar_tarea(progreso)
        return
    i = estado["paso_actual"] = pendientes[0]
    total = len(progreso["pasos"])
    anteriores = "\n\n".join(f"Paso {k + 1}: {p['resultado'][:300]}"
                              for k, p in enumerate(progreso["pasos"]) if p["estado"] == "hecho")
    # Todo lo que el trabajador sabe de la tarea sale del archivo de progreso
    pedido = (f"Tarea: {progreso['titulo']}\nObjetivo: {progreso['objetivo']}\n\n"
              f"Progreso:\n{progreso_como_texto(progreso)}\n\n"
              f"Resultados de los pasos anteriores (para no repetirlos):\n{anteriores or '(ninguno)'}\n\n"
              f"Paso a hacer ahora ({i + 1}/{total}): {progreso['pasos'][i]['paso']}")
    progreso["sesiones"] += 1
    guardar_progreso(progreso)
    try:
        _bucle_agente([{"role": "user", "content": pedido}], MODELO_TRABAJADOR, "trabajador",
                      mostrar=True, prefijo="  [trabajador] ")
    finally:
        estado["paso_actual"] = None
    progreso = leer_progreso()
    if progreso["pasos"][i]["estado"] != "hecho":
        print("\n  [tarea] El trabajador no registró el avance: el paso sigue pendiente.")
    print(f"\n  [tarea]\n{progreso_como_texto(progreso)}")
    if all(p["estado"] == "hecho" for p in progreso["pasos"]):
        entregar_tarea(progreso)


def entregar_tarea(progreso: dict) -> None:
    """Arma el documento final con los resultados de cada paso. Lo escribe el
    PROGRAMA, pero igual pide permiso, como cualquier acción con efectos (v18)."""
    if progreso.get("entregada"):
        print(f"\n(La tarea ya está terminada y entregada en {progreso['entregada']}.)\n")
        return
    nombre = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFD", progreso["titulo"].lower())
                    .encode("ascii", "ignore").decode()).strip("-")[:60] + ".md"
    documento = f"# {progreso['titulo']}\n\n" + "\n\n".join(p["resultado"] for p in progreso["pasos"]) + "\n"
    print("\n  [tarea] Están todos los pasos hechos.")
    if not pedir_confirmacion("escribir_archivo", {"nombre": nombre, "contenido": documento}):
        print("  [tarea] No se guardó. Escribí 'seguir' cuando quieras guardarlo.")
        return
    print(f"  [tarea] {escribir_archivo(nombre, documento)}")
    progreso["entregada"] = f"salida/{nombre}"
    guardar_progreso(progreso)


def derivar(destino: str, motivo: str) -> str:
    """El cambio de agente lo hace el bucle; aquí solo validamos y avisamos."""
    if destino not in EQUIPO:  # v23: hay agentes que no son del equipo de atención
        raise ValueError(f"No existe el agente '{destino}'")
    if destino == estado["agente"]:
        raise ValueError(f"Ya sos el agente '{destino}': respondé vos la consulta")
    return f"Conversación derivada a '{destino}' (motivo: {motivo})."


numeros_de_subagente = itertools.count(1)  # v22: para distinguir subagentes que corren a la vez


def delegar(tarea: str, modelo: str) -> str:
    """Corre un bucle de agente nuevo, con contexto vacío, y devuelve su resultado."""
    numero = next(numeros_de_subagente)
    memoria_subagente = [{"role": "user", "content": tarea}]  # contexto vacío
    resultado = _bucle_agente(
        memoria_subagente,
        MODELOS_SUBAGENTE[modelo],
        agente="subagente",
        mostrar=False,          # su texto no se muestra: es para el principal
        prefijo=f"    [subagente {numero}] ",
    )
    # Toda la memoria_subagente (búsquedas, fragmentos leídos...) se descarta aquí.
    return resultado or "(el subagente no devolvió texto)"


def ejecutar_herramienta(nombre: str, argumentos: dict) -> str:
    if nombre in MCP.destinos:  # herramienta de un servidor MCP (v19)
        return MCP.llamar(nombre, argumentos)
    if nombre == "crear_tarea":  # v23
        return crear_tarea(argumentos["titulo"], argumentos["pasos"])
    if nombre == "registrar_avance":  # v23
        return registrar_avance(argumentos["resultado"])
    if nombre == "actualizar_plan":  # v20
        return actualizar_plan(argumentos["pasos"])
    if nombre == "escribir_archivo":
        return escribir_archivo(argumentos["nombre"], argumentos["contenido"])
    if nombre == "enviar_mail":
        return enviar_mail(argumentos["para"], argumentos["asunto"], argumentos["cuerpo"])
    if nombre == "guardar_nota":
        return guardar_nota(argumentos["nota"])
    if nombre == "derivar":
        return derivar(argumentos["destino"], argumentos.get("motivo", ""))
    if nombre == "cargar_skill":
        return cargar_skill(argumentos["nombre"])
    if nombre == "delegar":
        return delegar(argumentos["tarea"], argumentos.get("modelo", "sonnet"))
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
    """Lee la conversación guardada y el agente que estaba atendiendo. Si el
    archivo está dañado, lo aparta y empieza de cero en lugar de caerse."""
    if not ARCHIVO_MEMORIA.exists():
        return []
    try:
        datos = json.loads(ARCHIVO_MEMORIA.read_text(encoding="utf-8"))
        estado["agente"] = datos["agente"] if datos["agente"] in AGENTES else AGENTE_INICIAL
        estado["plan"] = datos.get("plan", [])  # v20: el plan también se guarda
        return datos["mensajes"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        respaldo = ARCHIVO_MEMORIA.with_suffix(".danado.json")
        ARCHIVO_MEMORIA.rename(respaldo)
        print(f"(La memoria estaba dañada; la moví a {respaldo.name} y empiezo de cero)")
        return []


def guardar_memoria(memoria: list) -> None:
    datos = {"agente": estado["agente"], "plan": estado["plan"], "mensajes": memoria}  # v20: + plan
    ARCHIVO_MEMORIA.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
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
# Manejo del contexto (de la v11)
# ---------------------------------------------------------------------------

def tokens_usados(response) -> int:
    """Tamaño total de la última llamada: lo que entró (con y sin caché) más
    lo que salió. Es una buena medida de cuánto pesa ya la conversación."""
    u = response.usage
    return (
        u.input_tokens
        + (u.cache_creation_input_tokens or 0)
        + (u.cache_read_input_tokens or 0)
        + u.output_tokens
    )


def es_mensaje_del_usuario(mensaje: dict) -> bool:
    """True si es algo que escribió la persona (no un tool_result)."""
    return mensaje["role"] == "user" and isinstance(mensaje["content"], str)


def sin_thinking(mensajes: list) -> list:
    """Quita los bloques de razonamiento ("thinking") del historial.

    Cada bloque thinking viene firmado junto con todo lo que había antes en la
    conversación. Al resumir o recortar cambiamos ese "antes", así que las
    firmas dejan de valer y la API rechazaría esos bloques. Quitarlos una vez,
    justo al compactar, es lo recomendado; el resto del historial queda igual.
    """
    limpios = []
    for m in mensajes:
        if isinstance(m["content"], list):
            contenido = [b for b in m["content"]
                         if b.get("type") not in ("thinking", "redacted_thinking")]
            if not contenido:
                continue
            m = {**m, "content": contenido}
        limpios.append(m)
    return limpios


def punto_de_corte(memoria: list) -> int:
    """Índice desde donde se conservan los mensajes: el comienzo del
    antepenúltimo turno (o el que indique TURNOS_A_CONSERVAR). Como siempre es
    un mensaje del usuario, nunca separa un tool_use de su tool_result."""
    inicios = [i for i, m in enumerate(memoria) if es_mensaje_del_usuario(m)]
    if len(inicios) <= TURNOS_A_CONSERVAR:
        return 0
    return inicios[-TURNOS_A_CONSERVAR]


def como_texto(mensajes: list) -> str:
    """Pasa mensajes a texto plano para pedir el resumen."""
    lineas = []
    for m in mensajes:
        quien = "Usuario" if m["role"] == "user" else "Asistente"
        if isinstance(m["content"], str):
            lineas.append(f"{quien}: {m['content']}")
            continue
        for b in m["content"]:
            if b.get("type") == "text":
                lineas.append(f"{quien}: {b['text']}")
            elif b.get("type") == "tool_use":
                lineas.append(f"(El asistente usó {b['name']} con {json.dumps(b['input'], ensure_ascii=False)})")
            elif b.get("type") == "tool_result":
                contenido = str(b.get("content", ""))[:500]
                lineas.append(f"(Resultado de la herramienta: {contenido})")
    return "\n".join(lineas)


def resumir(mensajes: list) -> str:
    """Opción mejor: le pedimos a Claude un resumen de la parte vieja."""
    inicio = time.monotonic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},  # resumir no necesita pensar mucho
        system="Resumís conversaciones entre un usuario y un asistente. Conservá los datos "
               "concretos (nombres, números, decisiones, preferencias, fuentes citadas) y "
               "omití saludos y relleno. Escribí en español, en viñetas.",
        messages=[{"role": "user", "content": "Resumí esta conversación:\n\n" + como_texto(mensajes)}],
    )
    registrar_llamada(response, inicio)  # registro de la v09
    return "".join(b.text for b in response.content if b.type == "text").strip()


def compactar(memoria: list) -> list:
    """Reemplaza lo viejo por un resumen y deja intactos los últimos turnos."""
    corte = punto_de_corte(memoria)
    if corte == 0:
        return memoria  # todavía no hay nada viejo que resumir
    viejos, recientes = memoria[:corte], memoria[corte:]
    try:
        resumen = resumir(viejos)
        print(f"  [contexto] Resumí {len(viejos)} mensajes viejos.")
        cabeza = [{"role": "user", "content": f"[Resumen de la conversación anterior]\n{resumen}"}]
    except anthropic.APIError as e:
        # Opción simple (plan B): recortar sin resumir
        print(f"  [contexto] No pude resumir ({e.__class__.__name__}); recorto {len(viejos)} mensajes viejos.")
        cabeza = []
    return cabeza + sin_thinking(recientes)


# ---------------------------------------------------------------------------
# Router: elegir el modelo según la dificultad (de la v12)
# ---------------------------------------------------------------------------

def clasificar(pregunta: str, memoria: list) -> str:
    """Le pide a Haiku que clasifique la pregunta. Devuelve una clave de MODELOS.

    Además del mensaje nuevo le pasamos el mensaje anterior del usuario: sin
    ese contexto, una repregunta corta como "¿y para Lisboa?" parece trivial
    aunque continúe una consulta difícil."""
    anteriores = [m["content"] for m in memoria
                  if m["role"] == "user" and isinstance(m["content"], str)]
    contexto = f"Mensaje anterior del usuario: {anteriores[-1][:500]}\n\n" if anteriores else ""
    try:
        inicio = time.monotonic()
        response = client.messages.create(
            model=MODELO_CLASIFICADOR,
            max_tokens=1000,
            output_config={
                "effort": "low",
                # Structured outputs: la respuesta siempre es este JSON
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"dificultad": {"type": "string", "enum": list(MODELOS)}},
                        "required": ["dificultad"],
                        "additionalProperties": False,
                    },
                },
            },
            system="Clasificás mensajes que recibe un asistente de chat según cuánto "
                   "razonamiento necesita la respuesta. Si el mensaje continúa una "
                   "consulta anterior, clasificalo según la consulta completa:\n"
                   "- trivial: saludos, charla, datos simples, preguntar la hora.\n"
                   "- normal: preguntas concretas, buscar algo en documentos, una cuenta.\n"
                   "- dificil: problemas de varios pasos, análisis, comparar opciones, planificar.",
            messages=[{"role": "user", "content": f"{contexto}Mensaje a clasificar: {pregunta}"}],
        )
        registrar_llamada(response, inicio)  # registro de la v09
        texto = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(texto)["dificultad"]
    except (anthropic.APIError, json.JSONDecodeError, KeyError):
        return "normal"  # si el router falla, no frenamos la conversación


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

# v22: con herramientas en paralelo, varios hilos registran a la vez. El
# candado hace que escriban de a uno, así no se mezclan líneas ni se pierden sumas.
candado_registro = threading.Lock()


def registrar(evento: dict) -> None:
    """Agrega una línea JSON a registro.jsonl. Un archivo con un evento por
    línea se puede leer, filtrar y sumar después (con Python, una planilla o
    una herramienta de observabilidad)."""
    evento = {"momento": datetime.now().isoformat(timespec="seconds"),
              "programa": Path(__file__).stem, **evento}
    with candado_registro, open(ARCHIVO_REGISTRO, "a", encoding="utf-8") as archivo:
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
    with candado_registro:  # v22
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
    with candado_registro:  # v22
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

def llamar_claude(memoria: list, modelo: str, system: str, herramientas: list,
                  ultima_vuelta: bool = False, mostrar: bool = True, nombre: str = "Bot"):
    """Hace una llamada con streaming y devuelve el mensaje completo."""
    inicio = time.monotonic()  # registro (de la v09): para medir cuánto tarda
    with client.messages.stream(
        model=modelo,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=herramientas,
        # En la última vuelta permitida, Claude ya no puede pedir herramientas
        tool_choice={"type": "none"} if ultima_vuelta else {"type": "auto"},
        # Prompt caching: guarda el prefijo (herramientas + system + mensajes)
        # para que la próxima llamada lo lea de la caché.
        cache_control={"type": "ephemeral"},
        messages=memoria,
    ) as stream:
        hubo_texto = False
        for pedazo in stream.text_stream:
            if mostrar:
                if not hubo_texto:
                    print(f"\nBot ({nombre}): ", end="")
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

# Estado de la conversación:
#   tokens      (v11): tamaño de la última llamada, para decidir si compactar.
#   mostrar     (v13): si se imprimen los avisos (también los de los subagentes).
#   agente      (v15): qué agente está atendiendo al usuario.
#   acciones    (v18): acciones con efectos ya ejecutadas en el turno en curso.
#   interactivo (v18): si hay una persona para confirmar acciones; main() lo activa.
#   plan        (v20): los pasos de la tarea en curso.
#   objetivo_tarea, paso_actual (v23): la tarea larga que se está preparando y el
#                  paso que hace el trabajador de esta sesión (lo nuevo de esta versión).
estado = {"tokens": 0, "agente": AGENTE_INICIAL, "interactivo": False, "mostrar": True,
          "acciones": [], "plan": [], "objetivo_tarea": "", "paso_actual": None}


def turno(memoria: list, texto: str, mostrar: bool = True) -> str:
    """Clasifica la pregunta, agrega el mensaje del usuario, corre el bucle con
    el agente activo y devuelve el texto final. Si algo falla, deja la memoria
    (y el agente activo) como estaban antes del turno."""
    estado["mostrar"] = mostrar  # v13: así también callan los avisos de herramientas y subagentes
    dificultad = clasificar(texto, memoria)
    modelo = MODELOS[dificultad]
    if mostrar:
        print(f"  [router] {dificultad} -> {modelo}")
    inicio, agente_inicial = len(memoria), estado["agente"]
    plan_inicial = estado["plan"]  # v20: si el turno falla, el plan vuelve a como estaba
    estado["acciones"] = []  # v18: acciones con efectos ejecutadas en este turno
    memoria.append({"role": "user", "content": texto})
    try:
        respuesta = _bucle_agente(memoria, modelo, estado["agente"], mostrar)
        # v21: antes de dar por terminado el turno, otro modelo revisa la respuesta
        # De la v21: si hubo handoff, el bucle subió de Haiku a Sonnet (v15); la
        # corrección la hace el mismo especialista, así que también va con Sonnet.
        if estado["agente"] != agente_inicial and modelo == MODELOS["trivial"]:
            modelo = MODELOS["normal"]
        respuesta = verificar_y_corregir(memoria, inicio, texto, respuesta, modelo, mostrar)
        avisar_pasos_pendientes()  # v20: el programa revisa que no queden pasos colgados
        return respuesta
    except BaseException:
        del memoria[inicio:]  # deshacer el turno: nada de mensajes huérfanos
        estado["agente"] = agente_inicial
        estado["plan"] = plan_inicial
        # De la v18: lo que ya pasó en el mundo no se deshace. Si el
        # turno alcanzó a ejecutar acciones (un mail ya salió), las dejamos
        # anotadas en la memoria para que el bot no las repita creyendo que
        # no ocurrieron.
        if estado["acciones"]:
            registrar_acciones_hechas(memoria, texto)
        raise


def registrar_acciones_hechas(memoria: list, texto: str) -> None:
    """Deja en la memoria un turno corto que cuenta qué acciones se ejecutaron
    antes de que el turno fallara (de la v18)."""
    hechas = "; ".join(estado["acciones"])
    memoria.append({"role": "user", "content": texto})
    memoria.append({"role": "assistant", "content": [{"type": "text", "text":
        f"(Este turno se interrumpió por un error, pero antes llegué a ejecutar: {hechas}. "
        "No lo voy a repetir sin confirmarlo con el usuario.)"}]})
    guardar_memoria(memoria)  # ya mismo: main() no guarda los turnos que fallan
    if estado["mostrar"]:
        print(f"\n  [aviso] El turno falló, pero estas acciones ya se ejecutaron: {hechas}")


def _bucle_agente(memoria: list, modelo: str, agente: str,
                  mostrar: bool, prefijo: str = "  ") -> str:
    """El bucle del agente. Lo usan el equipo y los subagentes. En cada vuelta
    toma el system prompt y las herramientas del agente activo, que puede
    cambiar a mitad del turno si alguien usa derivar."""
    for vuelta in range(MAX_VUELTAS):
        ultima = vuelta == MAX_VUELTAS - 1
        config = AGENTES[agente]
        response = llamar_claude(memoria, modelo, config["system"], config["herramientas"],
                                 ultima_vuelta=ultima, mostrar=mostrar, nombre=agente)
        if agente in EQUIPO:  # solo la conversación principal cuenta para compactar (v23: EQUIPO)
            estado["tokens"] = tokens_usados(response)
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

        # v22: todas las herramientas de esta respuesta se ejecutan juntas (en
        # paralelo cuando se puede) y los resultados vuelven en el mismo orden.
        pedidos = [b for b in response.content if b.type == "tool_use"]
        resultados, nuevo_agente = ejecutar_pedidos(pedidos, prefijo), None
        motivo = ""
        for bloque, resultado in zip(pedidos, resultados):
            if bloque.name == "derivar" and not resultado["is_error"]:
                nuevo_agente = bloque.input["destino"]
                motivo = bloque.input.get("motivo", "")
        if nuevo_agente and nuevo_agente != agente:
            # De la v15: el que lee esto es el agente NUEVO. Le decimos quién es y que
            # responda ya; si no, ve en el historial el derivar del anterior,
            # cree que todavía es él y le anuncia el pase al usuario en vez de responder.
            resultados.append({"type": "text", "text":
                f"(Aviso del sistema: el agente '{agente}' te pasó la conversación. Ahora sos "
                f"'{nuevo_agente}' y los mensajes anteriores del asistente fueron de otro agente. "
                f"Lo que necesita el usuario: {motivo or 'ver su último mensaje'}. "
                "Respondé directamente esa consulta usando tus herramientas; no anuncies "
                "la derivación ni vuelvas a saludar.)"})
        if vuelta == MAX_VUELTAS - 2:
            # Avisamos a Claude que la próxima es su última oportunidad
            resultados.append({"type": "text", "text": "Llegaste al límite de herramientas: "
                               "respondé ahora con la información que tenés."})
        memoria.append({"role": "user", "content": resultados})

        # El handoff (de la v15): desde la próxima vuelta responde otro agente
        if nuevo_agente and nuevo_agente != agente:
            if estado["mostrar"]:
                print(f"{prefijo}[handoff] {agente} -> {nuevo_agente}")
            agente = estado["agente"] = nuevo_agente
            # De la v15: el router clasificó el mensaje pensando en triage ("¿qué paquetes
            # tenés?" le parece trivial), pero el especialista tiene que buscar en
            # documentos y usar herramientas: como mínimo, el modelo normal.
            if modelo == MODELOS["trivial"]:
                modelo = MODELOS["normal"]
                if estado["mostrar"]:
                    print(f"{prefijo}[router] handoff -> {modelo}")
            # Cambiaron el system prompt y las herramientas: las firmas de los
            # bloques thinking ya no valen, así que los quitamos (como al compactar en la v11).
            memoria[:] = sin_thinking(memoria)

    return ""  # no se llega aquí: la última vuelta no permite herramientas


class RespuestaRechazada(Exception):
    """Claude terminó con stop_reason == "refusal"."""


def resultado_herramienta(bloque, prefijo: str) -> dict:
    """Ejecuta una herramienta. Si falla (incluso un subagente entero), Claude
    recibe el error (is_error) en lugar de que se caiga el programa."""
    if bloque.name == "delegar" and estado["mostrar"]:
        print(f"{prefijo}[delegar] modelo={bloque.input.get('modelo')}: {bloque.input.get('tarea', '')[:80]}")
    # Humano en el bucle (de la v18): sin un "sí", la acción no ocurre
    if necesita_confirmacion(bloque.name) and not pedir_confirmacion(bloque.name, bloque.input):
        if estado["mostrar"]:
            print(f"{prefijo}[herramienta] {bloque.name}: el usuario no lo autorizó")
        return {"type": "tool_result", "tool_use_id": bloque.id, "is_error": True,
                "content": "El usuario NO autorizó esta acción, así que no se ejecutó."}
    inicio = time.monotonic()
    try:
        resultado = ejecutar_herramienta(bloque.name, bloque.input)
        es_error = False
    except Exception as e:
        resultado, es_error = f"Error: {e.__class__.__name__}: {e}", True
    registrar_herramienta(bloque, resultado, es_error, inicio)  # de la v09
    # Registro de acciones (de la v18): anotamos cada acción con efectos que se
    # ejecutó, por si después el turno falla (ver turno()).
    if not es_error and necesita_confirmacion(bloque.name):
        estado["acciones"].append(f"{bloque.name} {json.dumps(bloque.input, ensure_ascii=False)[:200]}")
    if estado["mostrar"]:
        resumen = resultado if len(resultado) < 100 else resultado[:100].replace("\n", " ") + "..."
        print(f"{prefijo}[herramienta] {bloque.name}({bloque.input}) -> {resumen}")
    return {"type": "tool_result", "tool_use_id": bloque.id, "content": resultado, "is_error": es_error}


def texto_de(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main() -> None:
    # Si hay notas nuevas desde la última sesión, el system prompt cambió y
    # las firmas de los bloques thinking guardados ya no valen: los quitamos.
    memoria = sin_thinking(cargar_memoria())

    estado["interactivo"] = True  # hay una persona del otro lado para confirmar
    print("Chatbot v23 (v22 + tareas largas). Escribí 'salir' para terminar, 'olvidar' para "
          "borrar la conversación, 'notas' para ver las notas, 'olvidar notas' para borrarlas, "
          "'plan' para ver el plan, o 'tarea <objetivo>', 'tarea', 'seguir' y 'olvidar tarea' "
          "para las tareas largas.")
    archivos = {frag["fuente"] for frag in INDICE}
    print(f"(Documentos indexados: {len(archivos)} archivos, {len(INDICE)} fragmentos, "
          f"búsqueda por {MODO_BUSQUEDA})")
    print(f"(Skills disponibles: {', '.join(SKILLS) or 'ninguna'})")
    if memoria:
        print(f"(Memoria cargada: {len(memoria)} mensajes de sesiones anteriores)")
    print(f"(Te atiende: {estado['agente']})")
    print(f"(Notas a largo plazo: {len(NOTAS)})")
    print(f"(Herramientas MCP: {', '.join(MCP.destinos) or 'ninguna'})")
    piden = [h for h in MCP.destinos if MCP.pide_confirmacion(h)]
    if piden:
        print(f"(Piden confirmación: {', '.join(piden)})")
    print()

    while True:
        try:
            user_input = input("Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("salir", "exit", "quit"):
            break
        # Lo nuevo de la v23: comandos de tareas largas. El estado vive en
        # progreso_tarea.json, así que si una sesión falla alcanza con repetir 'seguir'.
        comando = user_input.lower()
        if comando == "olvidar tarea":
            ARCHIVO_PROGRESO.unlink(missing_ok=True)
            print("\n(Tarea descartada)\n")
            continue
        if comando == "tarea" or comando.startswith("tarea ") or comando == "seguir":
            empezar_registro_de_turno()  # registro de la v09
            try:
                if comando == "tarea":
                    progreso = leer_progreso()
                    print("\n" + (progreso_como_texto(progreso) if progreso else
                                   "(No hay ninguna tarea. Empezá una con: tarea <objetivo>)") + "\n")
                    continue
                if comando == "seguir":
                    seguir_tarea()
                else:
                    empezar_tarea(user_input[len("tarea "):].strip())
            except (anthropic.APIError, KeyboardInterrupt) as e:
                print(f"\n(La sesión de trabajo se cortó: {e.__class__.__name__}. Lo ya "
                      "registrado está guardado; escribí 'seguir' para retomar.)\n")
                continue
            print(resumen_del_turno())
            print()
            continue
        if user_input.lower() == "plan":  # de la v20
            print("\n" + (plan_como_texto(estado["plan"]) or "(No hay ningún plan en curso)") + "\n")
            continue
        if user_input.lower() == "notas":
            print("\n" + ("\n".join(f"- {n}" for n in NOTAS) or "(No hay notas guardadas)") + "\n")
            continue
        if user_input.lower() == "olvidar notas":
            NOTAS.clear()
            ARCHIVO_NOTAS.unlink(missing_ok=True)
            print("\n(Notas borradas. Los agentes las dejan de ver en la próxima sesión.)\n")
            continue
        if user_input.lower() == "olvidar":
            memoria.clear()
            estado["agente"] = AGENTE_INICIAL
            estado["plan"] = []  # v20
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

        # Compactamos entre turnos (nunca en medio de un bucle de herramientas)
        if estado["tokens"] > LIMITE_CONTEXTO:
            print(f"  [contexto] La conversación ocupa {estado['tokens']} tokens; compacto.")
            memoria[:] = compactar(memoria)

        print(resumen_del_turno())  # de la v09: qué costó este turno
        guardar_memoria(memoria)
        print()


if __name__ == "__main__":
    try:
        main()
    finally:
        MCP.cerrar()
