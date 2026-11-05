"""Versión 1: el chatbot más simple del mundo. Sin memoria.

Lee tu mensaje, llama al LLM, imprime la respuesta, repite.
Cada mensaje se envía solo: el bot no recuerda nada de lo anterior.

Todavía no es un agente: solo conversa. Pasa a ser un agente en la v03,
cuando puede decidir usar herramientas.

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v01_conteo_tokens.py
"""

import anthropic

from contexto import MonitorContexto

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
MODEL = "claude-opus-5-5"
# Ventana de Opus 5.5 (no confundir con max_tokens, el tope de salida).
# https://platform.claude.com/docs/en/build-with-claude/context-windows
monitor = MonitorContexto(ventana=1_000_000)

print("Chatbot v01 (sin memoria). Escribí 'salir' para terminar.\n")

while True:
    user_input = input("Vos: ").strip()
    if user_input.lower() in ("salir", "exit", "quit"):
        break
    if not user_input:
        continue

    messages = [{"role": "user", "content": user_input}]
    print("Esto es lo que le mandamos al LLM:")
    print(messages)
    print("[contexto] Sin memoria: se envía solo tu mensaje actual.\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,  # tope de tokens de la respuesta, incluido el
                           # razonamiento interno que el modelo hace antes de contestar
        messages=messages,  # solo el mensaje actual
    )

    # La respuesta llega en "bloques". Antes del texto puede venir un bloque
    # de razonamiento ("thinking"), así que juntamos solo los de texto.
    texto = "".join(b.text for b in response.content if b.type == "text")
    print(f"\nBot: {texto}\n")
    monitor.mostrar(response.usage, mensajes=len(messages))
