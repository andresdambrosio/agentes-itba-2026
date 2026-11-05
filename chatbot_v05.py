"""Versión 5: agente con herramientas + memoria en disco + streaming.

Igual que la v04, pero la respuesta aparece palabra por palabra mientras
Claude la va generando, en vez de esperar a que termine toda.

Lo nuevo es client.messages.stream(...) en lugar de client.messages.create(...):
    - stream.text_stream nos entrega el texto en pedacitos a medida que llega.
    - stream.get_final_message() nos da al final la respuesta completa
      (igual que antes), que usamos para la memoria y las herramientas.

La memoria se guarda en memoria.json (el mismo archivo que usa la v04).

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v05.py
"""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path

import anthropic

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
MODEL = "claude-opus-5-5"
ARCHIVO_MEMORIA = Path(__file__).parent / "memoria.json"


# ---------------------------------------------------------------------------
# Herramientas: funciones normales de Python
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
    if nombre == "obtener_hora":
        return obtener_hora()
    if nombre == "calcular":
        return calcular(argumentos["expresion"])
    return f"Herramienta desconocida: {nombre}"


# ---------------------------------------------------------------------------
# Memoria en disco (de la v04)
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

print("Chatbot v05 (herramientas + memoria en disco + streaming). Escribí 'salir' para terminar u 'olvidar' para borrar la memoria.")
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
        # Lo nuevo de la v05: stream() en vez de create(). El texto llega en
        # pedacitos y lo mostramos apenas llega, en vez de esperar la respuesta entera.
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
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
                print(f"  [herramienta] {bloque.name}({bloque.input}) -> {resultado}")
                resultados.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": resultado,
                })
        memoria.append({"role": "user", "content": resultados})

    # Guardamos solo cuando el turno terminó completo (sin herramientas pendientes)
    guardar_memoria(memoria)
    print()  # el texto ya se mostró en vivo; solo dejamos una línea en blanco
