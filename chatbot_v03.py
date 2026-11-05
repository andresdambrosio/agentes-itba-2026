"""Versión 3: chatbot con memoria + herramientas (ahora sí, un agente).

Le damos al bot dos herramientas: una calculadora y un reloj.
Claude decide solo cuándo usarlas. El ciclo es:

    1. Le mandamos la conversación + la lista de herramientas.
    2. Si Claude quiere usar una herramienta, nos lo pide (stop_reason == "tool_use").
    3. Nosotros la ejecutamos y le devolvemos el resultado.
    4. Repetimos hasta que Claude responda con texto normal.

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v03.py
"""

import ast
import operator
from datetime import datetime

import anthropic

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
MODEL = "claude-opus-5-5"


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


# ---------------------------------------------------------------------------
# Descripción de las herramientas para Claude (nombre, para qué sirve, parámetros)
# ---------------------------------------------------------------------------

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
# Bucle principal
# ---------------------------------------------------------------------------

memoria = []

print("Chatbot v03 (memoria + herramientas). Escribí 'salir' para terminar u 'olvidar' para borrar la memoria.\n")

while True:
    user_input = input("Vos: ").strip()
    if user_input.lower() in ("salir", "exit", "quit"):
        break
    if user_input.lower() == "olvidar":
        memoria.clear()
        print("\n(Memoria borrada)\n")
        continue
    if not user_input:
        continue

    memoria.append({"role": "user", "content": user_input})

    # Bucle del agente: seguimos mientras Claude pida herramientas
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=HERRAMIENTAS,
            messages=memoria,
        )

        # Guardamos la respuesta completa (puede incluir texto + pedidos de herramientas)
        memoria.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break  # Claude ya tiene la respuesta final

        # Ejecutamos cada herramienta que pidió y le devolvemos los resultados
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

    texto = "".join(b.text for b in response.content if b.type == "text")
    print(f"\nBot: {texto}\n")
