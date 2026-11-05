# Pendientes

Ideas para las próximas versiones y registro de lo hecho.

**Estado (2026-11-04):** hay 23 versiones y todas pasan las pruebas con el Claude simulado. **Falta volver a probarlas con la API real** (ver Mantenimiento).

Nota sobre la numeración: el 4 de noviembre la serie se reorganizó dos veces. Primero se partió la antigua v08 en tres (robustez, caché y compactación); después se agregó el registro de cada paso como v09. En este archivo todas las referencias usan la numeración actual.

## Plan de versiones

### v08 a v11: Robustez, registro y manejo del contexto

Van primero porque son la base: con subagentes los bucles se multiplican, y también los puntos donde algo puede fallar.

- [x] **Robustez.** → `chatbot_v08.py`
  - Manejar errores de la API (red, límites de uso) sin que el programa se caiga.
  - Si un turno falla a mitad, no dejar la memoria con un mensaje del usuario sin respuesta.
  - Poner un tope de vueltas al bucle de herramientas para que no quede en un ciclo infinito.
  - Avisar cuando la respuesta se corta por `max_tokens` (`stop_reason == "max_tokens"`).
- [x] **Registro de cada paso** (tiempo, tokens, costo estimado) en `registro.jsonl`, con un resumen por turno. Va temprano para que cada idea siguiente se pueda medir. → `chatbot_v09.py`
- [x] **Prompt caching** para abaratar el reenvío del mismo historial en cada llamada. → `chatbot_v10.py`
- [x] **Compactación.** La memoria crece sin límite: cada llamada es más cara y en algún momento choca con el límite de contexto del modelo. → `chatbot_v11.py`
  - Opción simple: recortar los mensajes más viejos (cuidando no separar un `tool_use` de su `tool_result`).
  - Opción mejor: cuando la conversación pasa de cierto tamaño, pedirle a Claude que la resuma y reemplazar lo viejo por el resumen.

### v12: Selección de modelo (router)

- [x] → `chatbot_v12.py`. Antes de responder, un modelo barato (Haiku) clasifica la pregunta como trivial, normal o difícil, y según eso se usa Haiku, Sonnet u Opus.
- Muestra el equilibrio entre costo, velocidad y calidad, que ahora se puede medir con el registro de la v09.

### v13: Subagentes

- [x] → `chatbot_v13.py`. Una herramienta `delegar(tarea, modelo)` que por dentro corre **otro bucle de agente**, con su propio system prompt, sus herramientas y su contexto vacío. Devuelve solo el resultado.
- Ventaja principal: **aislar el contexto**. Si el subagente lee diez fragmentos de documentos, esos fragmentos no ensucian la memoria del agente principal.

### v14: Skills

- [x] → `chatbot_v14.py`. Una carpeta `skills/` con un archivo `.md` por tipo de tarea. El agente solo ve el nombre y una línea de descripción de cada skill, y carga el texto completo cuando lo necesita con `cargar_skill(nombre)`.

### v15: Handoffs

- [x] → `chatbot_v15.py`. Un agente de triage que deriva la conversación a agentes especialistas (reservas y facturación).
- Diferencia con un subagente, que es **quién habla con el usuario**: con un subagente el principal sigue hablando; con un handoff habla el otro.

## Otras ideas (v16 a v19)

- [x] **Memoria a largo plazo.** → `chatbot_v16.py` Una herramienta `guardar_nota` con la que el bot anota datos sueltos del usuario. Las notas se cargan en el system prompt al empezar cada sesión.
- [x] **RAG con embeddings.** → `chatbot_v17.py` (Voyage AI) y `python evaluar.py --busqueda` para comparar con la búsqueda por palabras clave.
- [x] **Herramientas que hacen cosas, con confirmación.** → `chatbot_v18.py` Escribir un archivo o mandar un mail, con permiso del usuario ("humano en el bucle").
- [x] **MCP (Model Context Protocol).** → `chatbot_v19.py` + `servidor_mcp_ejemplo.py`
- [x] **Evaluaciones.** → `evaluar.py` + `evaluaciones.json`

## Piezas del harness que faltaban (v20 a v23)

Surgieron de comparar la serie con lo que hoy se llama *harness*: todo el programa que rodea al modelo.

- [x] **Planificación.** → `chatbot_v20.py` Herramienta `actualizar_plan`: el agente escribe la lista de pasos y la va tachando. El programa la valida y avisa si quedan pasos sin hacer.
- [x] **Verificación antes de responder.** → `chatbot_v21.py` Un revisor (otro modelo, otro system prompt) compara la respuesta con los fragmentos consultados; si encuentra un error, el agente corrige. Patrón "evaluador y optimizador".
- [x] **Herramientas en paralelo.** → `chatbot_v22.py` Lo que Claude pide junto en una respuesta se ejecuta a la vez, con hilos. Patrón "orquestador y trabajadores".
- [x] **Tareas largas en varias sesiones.** → `chatbot_v23.py` Un agente inicializador divide la tarea y la guarda en `progreso_tarea.json`; en cada sesión, un trabajador con contexto vacío hace un paso. Sigue el enfoque que Anthropic describe para agentes de larga duración.

## Mantenimiento

- [x] README que explique qué agrega cada versión y qué es el harness.
- [x] Probar las versiones sin API: `python pruebas/probar_todas.py` (23/23 pasan con el Claude simulado).
- [x] Probar con la API real (2026-11-03, con la numeración y el manual anteriores). La prueba encontró dos problemas, ya corregidos:
  - v01 y v02 leían `response.content[0].text`, pero con Opus 5.5 el primer bloque es de razonamiento (`thinking`) y fallaban. El simulador ahora también devuelve ese bloque, así que detecta el error.
  - El agente de facturación (v15 en adelante) se presentaba como "Mateo, el especialista" y no daba el contacto humano del manual.
- [x] Puntajes con `evaluar.py` (2026-11-03, 20 preguntas, manual anterior): la actual v11 sacó 20/20 y la actual v19 llegó a 20/20 después de corregir el triage (que no sabía a quién pasar las consultas generales), `derivar` (que dejaba derivar al mismo agente) y `evaluar.py` (que dejaba notas de "Ana Pérez" en `notas.json`).
- [x] **Revisión del 2026-11-04.**
  - La vieja v08 se partió en robustez, caché y compactación, y se agregó el registro como v09.
  - v19: las herramientas MCP pasan por la confirmación, salvo las de solo lectura de servidores marcados `"confiable": true`.
  - v18: la confirmación muestra el contenido completo y, si un turno falla después de una acción, la memoria guarda constancia.
  - v03 en adelante: la calculadora limita exponentes y tamaño del resultado.
  - v07 en adelante: manual más largo, un fragmento por sección, palabras vacías fuera de la búsqueda y las reglas de la v06 en el system prompt.
  - v12 en adelante: el router también ve el mensaje anterior del usuario.
  - v13 en adelante: `mostrar=False` silencia todos los avisos.
  - v01–v07: `max_tokens` pasa de 1024 a 16000.
  - `evaluaciones.json`: 24 preguntas, variantes con `|` y `no_debe_contener`. `evaluar.py` ya no se corta si falla el juez.
  - Todo en español rioplatense (vos), también lo que dice el bot.
  - Se agregaron las versiones 20 a 23.
- [ ] **Volver a correr las pruebas con la API real** y `evaluar.py chatbot_v08.py chatbot_v17.py chatbot_v20.py chatbot_v21.py chatbot_v23.py`, y anotar los puntajes y el costo de cada una (sale de `registro.jsonl`).
- [ ] Correr `python evaluar.py --busqueda` con `VOYAGE_API_KEY` para tener el puntaje de los embeddings con el manual nuevo (por palabras clave: 17/24).
- [ ] Revisar los precios de `PRECIOS` (v09 en adelante) contra la página oficial cuando cambien.

## Ideas nuevas

- [ ] **Permisos como política y hooks.** La v18 pregunta herramienta por herramienta. Los harness reales suman reglas ("esto siempre sí, esto nunca, esto preguntar") y puntos donde correr código propio antes o después de cada herramienta.
- [ ] **Ejecutar código en un entorno aislado.** Que el agente escriba y corra programas en vez de tener solo herramientas fijas. Es lo más potente y lo más delicado de hacer con seguridad.
- [ ] **Fallback ante rechazos.** La v08 solo avisa cuando Claude rechaza una respuesta (`stop_reason == "refusal"`). La API puede reintentar sola con otro modelo (parámetro `fallbacks`, en beta), pero el historial necesita un tratamiento especial después de un fallback.
- [ ] **Compactación del lado del servidor.** La v11 resume a mano para que se vea cómo funciona; la API también puede hacerlo sola (beta `compact`). Comparar las dos.
- [ ] **Verificar también las tareas largas.** En la v23 el trabajador registra su resultado sin revisión; se podría pasar cada paso por el revisor de la v21.
