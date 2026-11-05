"""Versión 2: el mismo chatbot, ahora con memoria de la conversación.

La "memoria" es simplemente una lista con todos los mensajes anteriores
(tuyos y del bot). En cada vuelta se le envía la lista completa al LLM,
así puede responder teniendo en cuenta todo lo que se habló.
La memoria vive mientras el programa está abierto; al salir se pierde.

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v02.py
"""

import anthropic

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
MODEL = "claude-opus-5-5"
memoria = []  # historial de la conversación

print("Chatbot v02 (con memoria). Escribí 'salir' para terminar u 'olvidar' para borrar la memoria.\n")

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

    # 1. Guardamos lo que dijo el usuario
    memoria.append({"role": "user", "content": user_input})

    # 2. Enviamos TODA la conversación, no solo el último mensaje
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,  # tope de tokens de la respuesta, incluido el
                           # razonamiento interno que el modelo hace antes de contestar
        messages=memoria,
    )

    # 3. Guardamos la respuesta del bot para que la recuerde después
    #    (la respuesta llega en bloques; antes del texto puede venir un bloque
    #    de razonamiento, "thinking", así que juntamos solo los de texto)
    reply = "".join(b.text for b in response.content if b.type == "text")
    memoria.append({"role": "assistant", "content": reply})

    print(f"\nBot: {reply}\n")
