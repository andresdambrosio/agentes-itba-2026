"""Versión 16: v15 + memoria a largo plazo (notas).

La memoria de la conversación (v04) guarda TODO lo que se dijo, y por eso
termina resumida (v11) o borrada ("olvidar"). Pero hay datos que conviene
recordar siempre: "se llama Andrés", "prefiere respuestas cortas", "viaja
con una bicicleta".

Esta versión agrega notas sueltas que sobreviven a todo:

    - Una herramienta guardar_nota(nota) con la que el bot anota un dato
      cuando le parece importante para el futuro.
    - Las notas se guardan en notas.json.
    - Al empezar cada sesión, las notas se cargan en el system prompt de
      todos los agentes del equipo. Así el bot "conoce" al usuario aunque la
      conversación se haya borrado o compactado.

Las notas nuevas entran al system prompt en la PRÓXIMA sesión, no en la
actual: si el system prompt cambiara a mitad de la charla, se perdería la
caché (v10). En la sesión actual no hace falta, porque el dato ya está en el
historial.

Comandos nuevos:
    notas          -> muestra las notas guardadas.
    olvidar notas  -> borra las notas ("olvidar" solo borra la conversación).

Uso:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v16.py
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
MODEL = "claude-opus-5-5"  # el modelo por defecto (y el que hace los resúmenes)
MAX_TOKENS = 16000
CARPETA = Path(__file__).parent
ARCHIVO_MEMORIA = CARPETA / "memoria_v16.json"
ARCHIVO_REGISTRO = CARPETA / "registro.jsonl"     # v09: una línea por llamada y por herramienta
ARCHIVO_NOTAS = CARPETA / "notas.json"

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


# Herramientas básicas: las tienen el principal y los subagentes
HERRAMIENTAS_BASE = [
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
# Memoria a largo plazo: notas (lo nuevo de la v16)
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
    "reservas": {
        "system": con_notas(SYSTEM),
        "herramientas": HERRAMIENTAS_BASE + [HERRAMIENTA_DELEGAR, HERRAMIENTA_CARGAR_SKILL,
                                             HERRAMIENTA_DERIVAR, HERRAMIENTA_GUARDAR_NOTA],
    },
    "facturacion": {
        "system": con_notas(SYSTEM_FACTURACION),
        "herramientas": HERRAMIENTAS_BASE + [HERRAMIENTA_DERIVAR, HERRAMIENTA_GUARDAR_NOTA],
    },
    # No es parte del equipo: es el que usa delegar (v13). No puede derivar.
    "subagente": {
        "system": SYSTEM_SUBAGENTE,
        "herramientas": HERRAMIENTAS_BASE,
    },
}
AGENTE_INICIAL = "triage"


def derivar(destino: str, motivo: str) -> str:
    """El cambio de agente lo hace el bucle; aquí solo validamos y avisamos."""
    if destino not in AGENTES or destino == "subagente":
        raise ValueError(f"No existe el agente '{destino}'")
    if destino == estado["agente"]:
        raise ValueError(f"Ya sos el agente '{destino}': respondé vos la consulta")
    return f"Conversación derivada a '{destino}' (motivo: {motivo})."


def delegar(tarea: str, modelo: str) -> str:
    """Corre un bucle de agente nuevo, con contexto vacío, y devuelve su resultado."""
    memoria_subagente = [{"role": "user", "content": tarea}]  # contexto vacío
    resultado = _bucle_agente(
        memoria_subagente,
        MODELOS_SUBAGENTE[modelo],
        agente="subagente",
        mostrar=False,          # su texto no se muestra: es para el principal
        prefijo="    [subagente] ",
    )
    # Toda la memoria_subagente (búsquedas, fragmentos leídos...) se descarta aquí.
    return resultado or "(el subagente no devolvió texto)"


def ejecutar_herramienta(nombre: str, argumentos: dict) -> str:
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
        return datos["mensajes"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        respaldo = ARCHIVO_MEMORIA.with_suffix(".danado.json")
        ARCHIVO_MEMORIA.rename(respaldo)
        print(f"(La memoria estaba dañada; la moví a {respaldo.name} y empiezo de cero)")
        return []


def guardar_memoria(memoria: list) -> None:
    datos = {"agente": estado["agente"], "mensajes": memoria}
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
estado = {"tokens": 0, "agente": AGENTE_INICIAL, "mostrar": True}


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
    memoria.append({"role": "user", "content": texto})
    try:
        return _bucle_agente(memoria, modelo, estado["agente"], mostrar)
    except BaseException:
        del memoria[inicio:]  # deshacer el turno: nada de mensajes huérfanos
        estado["agente"] = agente_inicial
        raise


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
        if agente != "subagente":  # solo la conversación principal cuenta para compactar
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

        resultados, nuevo_agente = [], None
        motivo = ""
        for bloque in response.content:
            if bloque.type == "tool_use":
                resultado = resultado_herramienta(bloque, prefijo)
                resultados.append(resultado)
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
    inicio = time.monotonic()
    try:
        resultado = ejecutar_herramienta(bloque.name, bloque.input)
        es_error = False
    except Exception as e:
        resultado, es_error = f"Error: {e.__class__.__name__}: {e}", True
    registrar_herramienta(bloque, resultado, es_error, inicio)  # de la v09
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

    print("Chatbot v16 (v15 + memoria a largo plazo). Escribí 'salir' para terminar, 'olvidar' para "
          "borrar la conversación, 'notas' para ver las notas u 'olvidar notas' para borrarlas.")
    archivos = {frag["fuente"] for frag in INDICE}
    print(f"(Documentos indexados: {len(archivos)} archivos, {len(INDICE)} fragmentos)")
    print(f"(Skills disponibles: {', '.join(SKILLS) or 'ninguna'})")
    if memoria:
        print(f"(Memoria cargada: {len(memoria)} mensajes de sesiones anteriores)")
    print(f"(Te atiende: {estado['agente']})")
    print(f"(Notas a largo plazo: {len(NOTAS)})")
    print()

    while True:
        try:
            user_input = input("Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("salir", "exit", "quit"):
            break
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
    main()
