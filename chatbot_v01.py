"""Versión 1: el chatbot más simple del mundo. Sin memoria.

Lee tu mensaje, llama al LLM, imprime la respuesta, repite.
Cada mensaje se envía solo: el bot no recuerda nada de lo anterior.

Todavía no es un agente: solo conversa. Pasa a ser un agente en la v03,
cuando puede decidir usar herramientas.

Uso:
    pip install anthropic
    export ANTHROPIC_API_KEY=tu_clave
    python chatbot_v01.py
"""

import anthropic

client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno

print("Chatbot v01 (sin memoria). Escribí 'salir' para terminar.\n")

while True:
    user_input = input("Vos: ").strip()
    if user_input.lower() in ("salir", "exit", "quit"):
        break
    if not user_input:
        continue

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,  # tope de tokens de la respuesta, incluido el
                           # razonamiento interno que el modelo hace antes de contestar
        messages=[{"role": "user", "content": user_input}],  # solo el mensaje actual
    )
    messages = [{"role": "user", "content": user_input}]
    #print("Esto es lo que le mandamos al LLM:")
    #print(messages)
    #print("\n")

    # La respuesta llega en "bloques". Antes del texto puede venir un bloque
    # de razonamiento ("thinking"), así que juntamos solo los de texto.
    texto = "".join(b.text for b in response.content if b.type == "text")
    print(f"\nBot: {texto}\n")
