# Agentes: de chatbot a agente, paso a paso

Veintitrés versiones de un mismo chatbot en Python. Cada una agrega **una sola idea** sobre la anterior, y el código marca con comentarios qué cambió y por qué (`lo nuevo de la vN`; lo heredado dice `de la vN`). La idea es leerlas en orden y comparar cada una con la anterior, por ejemplo con `diff chatbot_v09.py chatbot_v10.py`.

Todas usan Claude a través de la librería oficial de Anthropic.

## Qué es el harness

Un agente es **el modelo más el programa que lo rodea**. A ese programa se le dice *harness*: el bucle que llama al modelo y ejecuta sus herramientas, el manejo del contexto y la memoria, los permisos, la recuperación de errores, el registro de lo que pasa y la coordinación entre varios agentes. El modelo es el mismo en las 23 versiones; lo que cambia, versión a versión, es el harness.

| Pieza del harness | Versiones |
|---|---|
| Bucle del agente y herramientas | v03, v07, v19 (MCP) |
| Memoria: de la sesión, en disco, a largo plazo | v02, v04, v16 |
| Contexto: system prompt, RAG, caché, compactación, skills | v06, v07, v10, v11, v14, v17 |
| Robustez y registro de cada paso | v08, v09 |
| Elección de modelo | v12 |
| Varios agentes: subagentes, handoffs, en paralelo | v13, v15, v22 |
| Permisos y humano en el bucle | v18, v19 |
| Planificación y verificación | v20, v21 |
| Tareas largas en varias sesiones | v23 |
| Evaluación | `evaluar.py` |

## Cómo ejecutarlas

Necesitás Python 3.10 o más nuevo (el `python3` que trae macOS es el 3.9 y no sirve). Desde la carpeta `Agentes`:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=tu_clave
python chatbot_v01.py        # o chatbot_v02.py, chatbot_v03.py, ... chatbot_v23.py
```

También podés usar `./correr.sh 5` (o `./correr.sh 05`): corre esa versión con el Python de `.venv` y la clave guardada en `.env`. Sin número, corre la última.

De la v01 a la v16 solo hace falta `anthropic`. La v17 usa `voyageai` (opcional) y desde la v19 se usa `mcp`.

Comandos dentro del chat:

- `salir`: termina el programa (todas las versiones).
- `olvidar`: borra la memoria (desde la v02; desde la v04 también borra el archivo).
- `notas` y `olvidar notas`: muestran o borran la memoria a largo plazo (desde la v16).
- `plan`: muestra el plan de la tarea en curso (desde la v20).
- `tarea <objetivo>`, `tarea`, `seguir` y `olvidar tarea`: tareas largas en varias sesiones (v23).

## Ver cómo se ocupa la ventana de contexto

Las variantes `chatbot_v01_conteo_tokens.py`, `chatbot_v02_conteo_tokens.py`
y `chatbot_v03_conteo_tokens.py` muestran después de cada llamada al modelo una tabla
con todas las llamadas de la sesión:
cuántos mensajes se enviaron, tokens de entrada, tokens de salida y porcentaje
de la ventana usado por la entrada más la salida de esa llamada. También aparece
una barra de ocupación para la última llamada; con mensajes cortos el porcentaje
es tan pequeño que la barra puede verse vacía.

```bash
./correr.sh chatbot_v01_conteo_tokens.py   # solo el mensaje actual
./correr.sh chatbot_v02_conteo_tokens.py   # incluye también el historial
./correr.sh chatbot_v03_conteo_tokens.py   # incluye historial y herramientas
```

Los archivos `chatbot_v01.py`, `chatbot_v02.py` y `chatbot_v03.py` conservan sus versiones originales.
En las variantes con conteo, para comparar, probá: `traducí mi nombre al portugués`, `Mi nombre es Andrés`
y `¿Qué te había pedido?`. En la v01 siempre se envía un mensaje; en la v02
se envían 1, 3, 5... y los tokens de entrada crecen. `olvidar` borra el historial
de la v02: la próxima llamada vuelve a enviar un mensaje. La tabla conserva
las mediciones de la sesión para que puedas ver esa caída.

En la v03 un mismo turno del usuario puede generar varias llamadas al modelo:
una para pedir herramientas y otra al recibir sus resultados. Cada llamada tiene
su propia fila. Los tokens de entrada incluyen las definiciones de herramientas
y los resultados que se envían. Probá `¿Cuánto es 234 por 17?` para verlo.

El monitor compartido está en `contexto.py`. Los conteos vienen de `response.usage`,
incluyen los tokens de caché si los hay y no requieren llamadas adicionales.
La salida incluye razonamiento, aunque no se imprima como texto. La ocupación es
por llamada: sumar todas las filas mide uso acumulado, no el contexto actual.
La tabla vive en la terminal y se pierde al cerrar; no recupera ejecuciones anteriores.

La ventana configurada para `claude-opus-5-5` es de 1.000.000 de tokens según la
[documentación de Anthropic](https://platform.claude.com/docs/en/build-with-claude/context-windows).
Si cambiás de modelo, ajustá `ventana` en el script. `max_tokens=16000` es solo el
tope de salida, no el tamaño de la ventana ni la cantidad realmente generada.

## Las versiones

| Versión | Archivo | Qué agrega |
|---|---|---|
| v01 | `chatbot_v01.py` | El bucle mínimo: leer, llamar al LLM, imprimir, repetir. Sin memoria: cada mensaje va solo. Todavía no es un agente. |
| v02 | `chatbot_v02.py` | **Memoria de la conversación.** Una lista con todos los mensajes que se reenvía completa en cada llamada. |
| v03 | `chatbot_v03.py` | **Herramientas (tool use).** Calculadora y reloj. Aparece el segundo bucle: Claude pide una herramienta, la ejecutamos, le devolvemos el resultado, y así hasta que responde. Desde acá es un agente. |
| v04 | `chatbot_v04.py` | **Memoria en disco.** La conversación se guarda en `memoria.json` y se recupera al volver a abrir el programa. |
| v05 | `chatbot_v05.py` | **Streaming.** La respuesta aparece palabra por palabra mientras se genera. |
| v06 | `chatbot_v06.py` | **System prompt.** Instrucciones fijas (personalidad y reglas) que van aparte de la conversación. |
| v07 | `chatbot_v07.py` | **RAG.** Una herramienta `buscar_documentos` que busca en los archivos de `documentos/` y responde citando la fuente. |
| v08 | `chatbot_v08.py` | **Robustez.** Errores de la API sin que el programa se caiga, deshacer un turno que falló, tope de vueltas al bucle y avisos de respuesta cortada o rechazada. El código pasa a funciones. |
| v09 | `chatbot_v09.py` | **Registro de cada paso.** Cada llamada y cada herramienta quedan en `registro.jsonl` (tiempo, tokens, costo estimado) y cada turno termina con un resumen. |
| v10 | `chatbot_v10.py` | **Prompt caching.** La API guarda el prefijo repetido de cada llamada (herramientas, system prompt e historial) y lo cobra mucho más barato. |
| v11 | `chatbot_v11.py` | **Compactación.** Cuando la conversación crece demasiado, Claude resume lo viejo y se conservan los últimos turnos. |
| v12 | `chatbot_v12.py` | **Router de modelos.** Haiku clasifica cada pregunta como trivial, normal o difícil y según eso responde Haiku, Sonnet u Opus. |
| v13 | `chatbot_v13.py` | **Subagentes.** Una herramienta `delegar(tarea, modelo)` que corre otro agente con contexto vacío y devuelve solo su resultado. |
| v14 | `chatbot_v14.py` | **Skills.** Instrucciones por tipo de tarea en `skills/`. El agente solo ve la lista y carga el texto completo con `cargar_skill` cuando lo necesita. |
| v15 | `chatbot_v15.py` | **Handoffs.** Un equipo de agentes (triage, reservas, facturación). La herramienta `derivar` pasa la conversación a otro agente, que desde ahí atiende al usuario. |
| v16 | `chatbot_v16.py` | **Memoria a largo plazo.** Con `guardar_nota` el bot anota datos sueltos del usuario, que se cargan en el system prompt al empezar cada sesión. |
| v17 | `chatbot_v17.py` | **RAG con embeddings.** La búsqueda pasa de palabras clave a búsqueda por significado (Voyage AI). |
| v18 | `chatbot_v18.py` | **Acciones con confirmación.** `escribir_archivo` y `enviar_mail`, que solo se ejecutan si el usuario las autoriza ("humano en el bucle"). |
| v19 | `chatbot_v19.py` | **MCP.** Herramientas de servidores MCP externos, configurados en `mcp_servidores.json`, sin programarlas una por una. |
| v20 | `chatbot_v20.py` | **Planificación.** Con `actualizar_plan` el agente escribe la lista de pasos de una tarea y la va tachando; el programa la valida y avisa si quedan pasos sin hacer. |
| v21 | `chatbot_v21.py` | **Verificación.** Antes de dar por terminada una respuesta basada en documentos, otro modelo la compara con los fragmentos consultados; si encuentra un error, el agente corrige. |
| v22 | `chatbot_v22.py` | **Herramientas en paralelo.** Lo que Claude pide junto en una respuesta (por ejemplo, un subagente por destino) se ejecuta a la vez. |
| v23 | `chatbot_v23.py` | **Tareas largas.** Un agente inicializador divide la tarea en pasos y la guarda en `progreso_tarea.json`; en cada sesión, un agente trabajador con contexto vacío hace un paso. |

### Notas por versión

- **v01 → v02:** la única diferencia es que guardamos los mensajes en una lista. Sin eso, el bot no puede seguir una conversación.
- **v01:** `max_tokens` es el tope de la respuesta e incluye el razonamiento interno que el modelo hace antes de contestar; por eso no conviene ponerlo muy bajo.
- **v03:** las herramientas son funciones normales de Python más una descripción (nombre, para qué sirve y qué parámetros recibe) que Claude lee para decidir cuándo usarlas. La calculadora no usa `eval()`, para no ejecutar código arbitrario, y limita exponentes y resultados: sin eso, `9**9**9` cuelga el programa.
- **v04:** las respuestas de Claude llegan como objetos de la librería; la función `a_dict` los convierte en diccionarios para poder guardarlos en JSON. La memoria se guarda solo cuando el turno terminó completo.
- **v05:** se cambia `client.messages.create(...)` por `client.messages.stream(...)`. Al final se usa `stream.get_final_message()` para tener la respuesta completa (para la memoria y las herramientas).
- **v06:** el system prompt va en el parámetro `system`, no dentro de los mensajes, y no se guarda en la memoria. Para cambiar la personalidad, editá la constante `SYSTEM`.
- **v07:** la búsqueda es por palabras clave (TF-IDF) en Python puro, sin librerías extra, e ignora las palabras vacías ("que", "con", "para"...). El manual se parte en un fragmento por sección (17 en total) y la búsqueda devuelve los 3 más relevantes. El system prompt conserva las reglas de la v06 (usar `calcular` y `obtener_hora`).
- **v08:** si un turno falla, se deshace entero; hay un tope de `MAX_VUELTAS` al bucle de herramientas; se avisa si la respuesta se corta por `max_tokens` o si Claude la rechaza (`refusal`); una herramienta que falla le devuelve el error a Claude (`is_error`) en vez de tirar el programa. Desde acá el código está en funciones (`turno`, `main`), así `evaluar.py` puede importar cada versión.
- **v09:** `registro.jsonl` tiene una línea JSON por evento, fácil de filtrar y sumar con Python o una planilla. El costo es una estimación con los precios de lista de noviembre de 2026 (constante `PRECIOS`); revisalos en la [página de precios](https://platform.claude.com/docs/en/about-claude/pricing) antes de sacar conclusiones. Va temprano a propósito: desde acá, cada idea nueva se puede medir.
- **v10:** `cache_control={"type": "ephemeral"}` le pide a la API que guarde el prefijo de la petición; el bot muestra `[caché] N tokens leídos de la caché` cuando lo aprovecha y el registro de la v09 muestra cómo baja el costo. La caché solo sirve si el prefijo es idéntico, y vence a los pocos minutos sin uso.
- **v11:** cuando la conversación pasa de `LIMITE_CONTEXTO` tokens, Claude resume lo viejo y se conservan intactos los últimos turnos. Si el resumen falla, se recortan los mensajes viejos. El corte siempre cae al comienzo de un mensaje del usuario, así nunca se separa un `tool_use` de su `tool_result`. Al compactar se quitan los bloques `thinking` del historial, porque su firma valida todo lo que había antes ([documentación](https://platform.claude.com/docs/build-with-claude/preserved-thinking)).
- **v12:** el clasificador usa *structured outputs* (`output_config.format` con un esquema JSON) para que Haiku conteste siempre una de las tres palabras. Además del mensaje nuevo recibe el mensaje anterior del usuario, para clasificar bien repreguntas como "¿y para Lisboa?". Cada modelo tiene su propia caché, así que cambiar de modelo entre turnos reduce el ahorro de la v10.
- **v13:** el subagente usa la misma función de bucle que el principal, con otro system prompt, las herramientas básicas (no puede delegar) y una memoria nueva. Lo que lee queda en su memoria, que se descarta al terminar: al principal solo le llega el resultado.
- **v14:** cada skill es un `.md` con un encabezado `nombre` / `descripcion`. El system prompt se arma una sola vez al arrancar, para no romper la caché.
- **v15:** un handoff es cambiar a mitad de la conversación el system prompt y las herramientas activas. El agente activo se guarda con la memoria. Diferencia con la v13: con un subagente el principal sigue hablando; con un handoff habla el otro.
- **v16:** las notas nuevas entran al system prompt en la *próxima* sesión, para no invalidar la caché de la actual. Las notas sobreviven a `olvidar` y a la compactación.
- **v17:** los embeddings de los fragmentos se guardan en `embeddings_cache.json`. Sin `VOYAGE_API_KEY` la v17 vuelve a la búsqueda por palabras clave.
- **v18:** la confirmación la pide el *programa*, no Claude: aunque el modelo se equivoque, la acción no ocurre sin un "sí". El programa muestra el contenido completo de lo que se va a hacer, sin recortes. Si un turno falla después de ejecutar una acción, la memoria guarda una constancia (un mail enviado no se deshace, y el bot no debe mandarlo de nuevo). Cuando no hay nadie para confirmar (por ejemplo, en `evaluar.py`), la respuesta es siempre no. El mail es simulado: se guarda como `.eml` en `bandeja_salida/`.
- **v19:** las conexiones MCP viven en un hilo aparte con su propio event loop, porque la librería `mcp` es asíncrona y el resto del programa no. `servidor_mcp_ejemplo.py` ofrece `dias_hasta` y `convertir_moneda` (tasas ficticias), de solo lectura, y `guardar_recordatorio`, que escribe un archivo. Las herramientas MCP pasan por la confirmación de la v18, salvo las que el servidor declara de solo lectura y solo si en `mcp_servidores.json` ese servidor tiene `"confiable": true` (esas declaraciones las escribe el propio servidor; de uno desconocido no hay que creerles). También hay un ejemplo desactivado del servidor oficial de archivos (necesita Node.js); se activa cambiando `"activo"` a `true`.
- **v20:** el agente manda siempre el plan completo, así el programa no tiene que llevar la cuenta de qué cambió. El programa valida el plan (un solo paso `en_curso` a la vez, estados válidos) y, al terminar el turno, avisa si quedaron pasos sin hacer: un harness no confía en que el modelo se acuerde de todo, lo comprueba. Diferencia con las skills de la v14: una skill la escribió alguien de antemano; el plan lo arma el agente para esta tarea.
- **v21:** es el patrón "evaluador y optimizador". El revisor es otro agente, con su propio system prompt, que solo ve la pregunta, los fragmentos y la respuesta: así no se deja convencer por el razonamiento del que respondió. Solo revisa respuestas que usaron documentos (si no, no hay contra qué comparar) y como máximo `MAX_REVISIONES` veces por turno. Diferencia con `evaluar.py`: la evaluación mide después y con preguntas de prueba; la verificación actúa en cada respuesta real, antes de que llegue al cliente.
- **v22:** es el patrón "orquestador y trabajadores". Las herramientas van en hilos (`ThreadPoolExecutor`); las que piden confirmación o cambian el estado del programa (`derivar`, `guardar_nota`, `actualizar_plan`) siguen yendo de a una. El registro usa un candado (`threading.Lock`) para que los hilos no se pisen.
- **v23:** la memoria de trabajo es el archivo, no la conversación: se puede cerrar el programa entre paso y paso, o puede fallar una sesión, y alcanza con volver a escribir `seguir`. El programa decide qué paso le toca a cada trabajador, así el modelo no puede marcar como hecho uno que no hizo. Al final el programa arma el documento con los resultados y pide permiso para guardarlo en `salida/`. Diferencias: el plan de la v20 vive en una conversación y el progreso en un archivo; las notas de la v16 son datos del usuario y el progreso es el estado de un trabajo.

## Evaluaciones

`evaluar.py` le hace a una o varias versiones las preguntas de `evaluaciones.json` y cuenta cuántas responde bien. Sirve para comprobar si un cambio mejora o empeora el bot, en lugar de juzgarlo a ojo.

```bash
python evaluar.py --busqueda                          # palabras clave vs. embeddings, sin usar Claude
python evaluar.py chatbot_v08.py chatbot_v17.py        # corrige buscando datos clave en la respuesta
python evaluar.py --juez chatbot_v08.py chatbot_v17.py # corrige otro modelo (Sonnet) contra la respuesta de referencia
```

Solo funciona con la v08 en adelante. Los dos últimos modos usan la API y cuestan dinero: 24 preguntas por versión (más 24 correcciones con `--juez`). Desde la v09 el costo de cada versión queda en `registro.jsonl`. Comparar la v20 con la v21 muestra cuánto ayuda la verificación.

En la corrección por datos clave, cada pregunta tiene `debe_contener` (un dato con `|` acepta variantes, por ejemplo `"no hay reembolso|sin reembolso"`) y, a veces, `no_debe_contener`, para que una respuesta equivocada que menciona el número correcto no pase.

Con la búsqueda por palabras clave, el fragmento correcto queda primero en 17 de las 24 preguntas. Falla justo en las que usan otras palabras ("valijas" contra "valija", "quejarme" contra "reclamos", "mi gata" contra "perros y gatos"), que es lo que deberían resolver los embeddings.

Los puntajes con la API real que figuran en [PENDIENTES.md](PENDIENTES.md) son anteriores a la reorganización del 4 de noviembre de 2026. Hay que volver a correrlos.

## Pruebas sin API

`pruebas/` tiene un Claude simulado que permite probar el código de las 23 versiones sin clave y sin costo:

```bash
python pruebas/probar_todas.py         # todas las versiones
python pruebas/probar_todas.py 8 12    # solo algunas
```

Comprueba que el código funcione (el bucle, las herramientas, la memoria, los errores, la compactación, la confirmación, el plan, la verificación, el paralelismo y las tareas largas), no la calidad de las respuestas. Cada versión corre en una carpeta temporal, así que no toca tus archivos.

## Archivos que genera

- `memoria.json`: memoria compartida por v04, v05 y v06.
- `memoria_v07.json` … `memoria_v23.json`: memoria de cada versión desde la v07. Van aparte porque cada una tiene herramientas distintas y, desde la v15, también guardan el agente activo (y desde la v20, el plan).
- `registro.jsonl`: una línea por llamada al modelo y por herramienta (v09 en adelante).
- `notas.json`: memoria a largo plazo (v16 en adelante).
- `embeddings_cache.json`: embeddings ya calculados (v17 en adelante).
- `salida/` y `bandeja_salida/`: archivos y mails que el bot creó con permiso (v18 en adelante).
- `recordatorios.txt`: lo que guarda la herramienta MCP `guardar_recordatorio` (v19 en adelante).
- `mcp_servidores.log`: mensajes de los servidores MCP (v19 en adelante).
- `progreso_tarea.json`: la tarea larga en curso (v23).

Podés borrarlos cuando quieras (o usar `olvidar` dentro del chat).

## Documentos para el RAG

La carpeta `documentos/` trae `manual_nimbo_viajes.md`, el manual de una empresa ficticia, con unas quince secciones. Como la empresa no existe, Claude solo puede responder sobre ella si busca en el documento. Preguntas para probar:

- ¿Cuánto me devuelven si cancelo 20 días antes del viaje?
- ¿Cuántos puntos necesito para una noche de hotel gratis?
- ¿Puedo llevar a mi gata en el avión?

Para usar tus propios documentos, copiá archivos `.txt` o `.md` a `documentos/` y reiniciá el programa.

Para probar las versiones más nuevas:

- **Robustez (v08):** cortá internet y mandá un mensaje: el chat avisa y sigue.
- **Registro (v09):** mandá un mensaje y mirá la línea `[registro]`; después abrí `registro.jsonl`.
- **Caché (v10):** mandá dos mensajes seguidos y compará el costo de los dos turnos.
- **Router (v12):** "¿Cuánto cuesta una valija extra?" y después "¿y si son dos, ida y vuelta?".
- **Skills (v14):** "Quiero presentar un reclamo: el hotel de Bariloche no tenía mi reserva."
- **Handoffs (v15):** escribí "hola" y después "¿cuántos puntos Nube tengo que juntar para un pasaje?"
- **Notas (v16):** "Me llamo Ana y viajo siempre con mi bicicleta". Después cerrá el programa, abrilo y escribí `notas`.
- **Confirmación (v18):** "Guardame en un archivo un resumen de la política de cancelación."
- **MCP (v19):** "Si viajo el 2026-12-20 y cancelo hoy, ¿cuánto me devuelven?" y "Anotame un recordatorio para el 2026-12-15: pagar el saldo" (este pide confirmación).
- **Plan (v20):** "Somos 12, viajamos a Lisboa en febrero con dos valijas extra cada uno y un perro de 6 kg: armame el presupuesto." Después escribí `plan`.
- **Verificación (v21):** cualquier pregunta sobre el manual; mirá la línea `[verificación]`.
- **Paralelo (v22):** "Compará Bariloche, Cartagena y Lisboa: para cada uno averiguá promociones y qué incluye." Mirá la línea `[paralelo]`.
- **Tareas largas (v23):** `tarea Armar una guía para clientes nuevos con una sección por tema del manual`, después `seguir` varias veces (podés cerrar el programa en el medio).

## Próximos pasos

Las ideas y su estado están en [PENDIENTES.md](PENDIENTES.md).
