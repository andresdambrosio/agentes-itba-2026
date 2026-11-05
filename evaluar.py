"""Evaluaciones: medir automáticamente si una versión responde mejor que otra.

Probar el bot "a mano" engaña: uno recuerda las respuestas buenas y se
olvida de las malas. Una evaluación es un conjunto fijo de preguntas con su
respuesta esperada (evaluaciones.json) que se le hace a cada versión, y un
puntaje al final. Si cambiás algo y el puntaje baja, lo ves enseguida.

Tres modos:

    python evaluar.py --busqueda
        Compara solo la BÚSQUEDA del RAG: palabras clave (v07) contra
        embeddings (v17). Para cada pregunta se fija si el primer fragmento que
        devuelve contiene el texto del manual que responde la pregunta.
        No usa la API de Claude (los embeddings sí necesitan VOYAGE_API_KEY).

    python evaluar.py chatbot_v08.py chatbot_v17.py
        Le hace todas las preguntas a cada versión (de la v08 en adelante),
        cada una con la memoria vacía, y revisa si la respuesta contiene los
        datos esperados ("debe_contener"; un dato con "|" acepta cualquiera de
        sus variantes, por ejemplo "no hay reembolso|sin reembolso") y que no
        contenga los prohibidos ("no_debe_contener"). Usa la API: cuesta dinero.

    python evaluar.py --juez chatbot_v08.py chatbot_v17.py
        Igual, pero la corrección la hace otro modelo (un "juez") comparando
        la respuesta con la respuesta de referencia. Es más flexible que
        buscar palabras, a cambio de una llamada más por pregunta.

Solo funciona con la v08 en adelante: desde ahí el código está en funciones
y se puede importar sin que arranque el chat.

Antes de correr los modos con la API, conviene probar el código sin gastar
nada: python pruebas/probar_todas.py

Desde la v09, cada llamada queda en registro.jsonl: después de evaluar se
puede ver cuánto costó cada versión. Desde la v21 la respuesta pasa por el
revisor antes de corregirse, así que comparar la v20 con la v21 muestra
cuánto ayuda la verificación.
"""

import contextlib
import importlib.util
import io
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

CARPETA = Path(__file__).parent
ARCHIVO_EVALUACIONES = CARPETA / "evaluaciones.json"
MODELO_JUEZ = "claude-sonnet-5-5"


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin separadores de miles ("5.000" -> "5000")."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", texto)


@contextlib.contextmanager
def notas_apartadas():
    """Aparta notas.json mientras dura la evaluación (v16+) y lo deja como estaba.
    Así tus notas no influyen en el puntaje y las que guarde el bot durante la
    evaluación (por ejemplo, "se llama Ana Pérez") no quedan en tu memoria."""
    notas = CARPETA / "notas.json"
    copia = notas.read_bytes() if notas.exists() else None
    notas.unlink(missing_ok=True)
    try:
        yield
    finally:
        if copia is None:
            notas.unlink(missing_ok=True)
        else:
            notas.write_bytes(copia)


def cargar_version(ruta: str):
    """Importa un chatbot_vN.py como módulo, sin ejecutar su main()."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = CARPETA / ruta
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):  # silenciar mensajes al importar
        spec.loader.exec_module(modulo)
    if not hasattr(modulo, "turno"):
        sys.exit(f"{ruta.name} no tiene la función turno(): solo se pueden evaluar la v08 en adelante.")
    return modulo


# ---------------------------------------------------------------------------
# Modo 1: comparar la búsqueda (palabras clave vs. embeddings)
# ---------------------------------------------------------------------------

def evaluar_busqueda(casos: list) -> None:
    v17 = cargar_version("chatbot_v17.py")
    modos = ["palabras"] + (["significado"] if v17.MODO_BUSQUEDA == "significado" else [])
    if len(modos) == 1:
        print("(Sin VOYAGE_API_KEY solo se evalúa la búsqueda por palabras clave.)\n")

    aciertos = {m: 0 for m in modos}
    for caso in casos:
        fila = []
        for modo in modos:
            primero = v17.buscar_documentos(caso["pregunta"], cantidad=1, modo=modo)
            ok = normalizar(caso["en_documento"]) in normalizar(primero)
            aciertos[modo] += ok
            fila.append(f"{modo}: {'OK ' if ok else 'MAL'}")
        print(f"  {' | '.join(fila)}  <- {caso['pregunta']}")

    print()
    for modo in modos:
        print(f"Búsqueda por {modo}: {aciertos[modo]}/{len(casos)} con el fragmento correcto en primer lugar")


# ---------------------------------------------------------------------------
# Modo 2: evaluar versiones completas del bot
# ---------------------------------------------------------------------------

def contiene(respuesta: str, dato: str) -> bool:
    """True si la respuesta contiene el dato o alguna de sus variantes
    separadas por "|" (las respuestas no siempre usan las mismas palabras)."""
    return any(normalizar(v) in normalizar(respuesta) for v in dato.split("|"))


def corregir_con_juez(cliente, caso: dict, respuesta: str) -> tuple[bool, str]:
    """Un modelo compara la respuesta con la de referencia."""
    resultado = cliente.messages.create(
        model=MODELO_JUEZ,
        max_tokens=2000,
        output_config={
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "correcta": {"type": "boolean"},
                        "motivo": {"type": "string"},
                    },
                    "required": ["correcta", "motivo"],
                    "additionalProperties": False,
                },
            },
        },
        system="Corregís respuestas de un asistente. Una respuesta es correcta si contiene los "
               "datos de la respuesta de referencia y no dice nada que la contradiga. No importa "
               "el estilo ni que agregue información extra que sea cierta.",
        messages=[{"role": "user", "content":
                   f"Pregunta: {caso['pregunta']}\n\nRespuesta de referencia: {caso['respuesta']}\n\n"
                   f"Respuesta a corregir: {respuesta}"}],
    )
    texto = "".join(b.text for b in resultado.content if b.type == "text")
    datos = json.loads(texto)
    return bool(datos["correcta"]), datos["motivo"]


def evaluar_version(ruta: str, casos: list, juez: bool) -> dict:
    modulo = cargar_version(ruta)
    nombre = Path(ruta).stem
    print(f"\n=== {nombre}")
    aciertos, errores, segundos = 0, 0, 0.0
    try:
        for caso in casos:
            memoria = []  # cada pregunta empieza de cero
            if hasattr(modulo, "AGENTE_INICIAL"):  # v15+: empezar siempre en triage
                modulo.estado["agente"] = modulo.AGENTE_INICIAL
            if "plan" in getattr(modulo, "estado", {}):  # v20+: sin el plan de la pregunta anterior
                modulo.estado["plan"] = []
            inicio = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    respuesta = modulo.turno(memoria, caso["pregunta"], mostrar=False)
            except Exception as e:
                errores += 1
                print(f"  ERROR  {caso['pregunta']}  ({e.__class__.__name__}: {e})")
                continue
            segundos += time.time() - inicio

            if juez:
                try:
                    ok, motivo = corregir_con_juez(modulo.client, caso, respuesta)
                except Exception as e:  # si falla el juez, no perdemos toda la evaluación
                    errores += 1
                    print(f"  ERROR  {caso['pregunta']}  (juez: {e.__class__.__name__}: {e})")
                    continue
            else:
                faltan = [d for d in caso["debe_contener"] if not contiene(respuesta, d)]
                sobran = [d for d in caso.get("no_debe_contener", []) if contiene(respuesta, d)]
                ok = not faltan and not sobran
                motivo = "; ".join(([f"falta: {', '.join(faltan)}"] if faltan else [])
                                   + ([f"no debería decir: {', '.join(sobran)}"] if sobran else []))
            aciertos += ok
            print(f"  {'OK ' if ok else 'MAL'}  {caso['pregunta']}" + (f"  ({motivo})" if not ok else ""))
    finally:
        if hasattr(modulo, "MCP"):  # v19: cerrar los servidores MCP
            modulo.MCP.cerrar()
    respondidas = len(casos) - errores
    return {
        "version": nombre,
        "aciertos": aciertos,
        "total": len(casos),
        "errores": errores,
        "segundos_promedio": segundos / respondidas if respondidas else 0,
    }


def main() -> None:
    argumentos = sys.argv[1:]
    casos = json.loads(ARCHIVO_EVALUACIONES.read_text(encoding="utf-8"))

    if "--busqueda" in argumentos:
        evaluar_busqueda(casos)
        return

    juez = "--juez" in argumentos
    versiones = [a for a in argumentos if not a.startswith("--")]
    if not versiones:
        print(__doc__)
        return

    llamadas = len(casos) * len(versiones)
    print(f"Se harán {llamadas} conversaciones con la API de Claude"
          + (" (más una corrección del juez por cada una)" if juez else "") + ".")

    with notas_apartadas():
        resultados = [evaluar_version(v, casos, juez) for v in versiones]

    print("\n=== Resumen")
    for r in resultados:
        print(f"  {r['version']:<14} {r['aciertos']:>2}/{r['total']} correctas, "
              f"{r['errores']} errores, {r['segundos_promedio']:.1f} s por respuesta")


if __name__ == "__main__":
    main()
