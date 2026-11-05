"""Prueba todas las versiones con el Claude falso de simulador.py.

    python pruebas/probar_todas.py          # todas
    python pruebas/probar_todas.py 8 12     # solo algunas

Cada versión corre en una carpeta temporal (así no toca tus memoria*.json ni
tus notas) con una conversación guionada que ejercita lo que agrega esa
versión. Una versión pasa si termina sin errores y su salida contiene lo
esperado. No usa la API ni necesita clave.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROYECTO = Path(__file__).resolve().parent.parent
SIMULADOR = Path(__file__).resolve().parent / "simulador.py"

CALCULO = '[usa:calcular {"expresion":"(234*17)/3"}]'
BUSQUEDA = '[usa:buscar_documentos {"consulta":"cancelo reembolso"}]'

# versión -> (archivo, entradas, textos que deben aparecer en la salida
#             [, textos que NO deben aparecer])
LARGO = "x" * 120_000  # un mensaje enorme, para que la v11 tenga que compactar
# v23: el inicializador crea dos pasos; cada paso trae la marca que hará el trabajador
_PASOS = ["Resumir equipaje [usa:registrar_avance " + json.dumps({"resultado": "## Equipaje\nValija de 23 kg."}) + "]",
          "Resumir mascotas [usa:registrar_avance " + json.dumps({"resultado": "## Mascotas\nHasta 8 kg en cabina."}) + "]"]
TAREA = ("tarea Guía para clientes nuevos [usa:crear_tarea "
         + json.dumps({"titulo": "Guía para clientes nuevos", "pasos": _PASOS}, ensure_ascii=False) + "]")
CONTENIDO_LARGO = "hola " * 100 + "FIN-DEL-CONTENIDO"  # la v18 debe mostrarlo entero
CASOS = {
    1: ("chatbot_v01.py", ["hola"], ["Respuesta simulada a: hola"]),
    2: ("chatbot_v02.py", ["hola", "olvidar"], ["Respuesta simulada", "Memoria borrada"]),
    3: ("chatbot_v03.py", [CALCULO, '[usa:obtener_hora {}]', '[usa:calcular {"expresion":"9**9**9"}]'],
        ["-> 1326.0", "obtener_hora", "Exponente demasiado grande"]),
    4: ("chatbot_v04.py", [CALCULO], ["-> 1326.0"]),
    5: ("chatbot_v05.py", [CALCULO], ["-> 1326.0"]),
    6: ("chatbot_v06.py", [CALCULO], ["-> 1326.0"]),
    7: ("chatbot_v07.py", [BUSQUEDA], ["[Fuente: manual_nimbo_viajes.md]", "17 fragmentos"]),
    8: ("chatbot_v08.py", [CALCULO, "[falla] hola", "[corta]", "[rechaza]", '[usa:no_existe {}]'],
        ["-> 1326.0", "No hay conexión", "max_tokens", "prefirió no responder", "Herramienta desconocida"]),
    9: ("chatbot_v09.py", ["hola", CALCULO], ["[registro] 1 llamadas, 0 herramientas", "[registro] 2 llamadas, 1 herramientas"]),
    10: ("chatbot_v10.py", ["hola", "sigo"], ["[caché]"]),
    11: ("chatbot_v11.py", ["hola", "otra", LARGO, "sigo"], ["[contexto] Resumí"]),
    12: ("chatbot_v12.py", ["hola"], ["[router] trivial -> claude-haiku-5-5"]),
    13: ("chatbot_v13.py",
         ['[usa:delegar {"tarea":"[usa:calcular {\\"expresion\\":\\"3*3\\"}]","modelo":"haiku"}]'],
         ["[subagente] [herramienta] calcular", "-> 9"]),
    14: ("chatbot_v14.py", ['[usa:cargar_skill {"nombre":"redactar_reclamo"}]'],
         ["Cómo redactar un reclamo formal"]),
    15: ("chatbot_v15.py", ['[usa:derivar {"destino":"facturacion","motivo":"reembolso"}]', "sigo"],
         ["[handoff] triage -> facturacion", "Bot (facturacion)"]),
    16: ("chatbot_v16.py", ['[usa:guardar_nota {"nota":"Se llama Ana"}]', "notas"],
         ["Nota guardada", "- Se llama Ana"]),
    17: ("chatbot_v17.py", [BUSQUEDA], ["búsqueda por palabras", "[Fuente: manual_nimbo_viajes.md]"]),
    18: ("chatbot_v18.py",
         ['[usa:escribir_archivo {"nombre":"a.md","contenido":"' + CONTENIDO_LARGO + '"}]', "s",
          '[usa:enviar_mail {"para":"a@b.com","asunto":"x","cuerpo":"y"}]', "n",
          '[usa:enviar_mail {"para":"a@b.com","asunto":"x","cuerpo":"y"}] [falla-despues]', "s", "sigo"],
         ["Archivo guardado en salida/a.md", "FIN-DEL-CONTENIDO", "el usuario no lo autorizó",
          "estas acciones ya se ejecutaron: enviar_mail"]),
    19: ("chatbot_v19.py",
         ['[usa:utilidades__convertir_moneda {"monto":100,"de":"EUR","a":"USD"}]',
          '[usa:utilidades__guardar_recordatorio {"fecha":"2026-12-20","texto":"pagar"}]', "n"],
         ["108.00 USD", "Piden confirmación: utilidades__guardar_recordatorio",
          "utilidades__guardar_recordatorio: el usuario no lo autorizó"],
         ["quiere usar utilidades__convertir_moneda"]),
    20: ("chatbot_v20.py",
         ['[usa:actualizar_plan {"pasos":[{"paso":"Buscar promos","estado":"hecho"},{"paso":"Comparar","estado":"pendiente"}]}]',
          '[usa:actualizar_plan {"pasos":[{"paso":"A","estado":"en_curso"},{"paso":"B","estado":"en_curso"}]}]', "plan"],
         ["[plan]", "[x] Buscar promos", "Pasos sin terminar (1)", "Solo un paso puede estar en_curso"]),
    21: ("chatbot_v21.py", [BUSQUEDA + " [verifica-mal]", BUSQUEDA],
         ["[verificación] Problema", "Le pido al agente que corrija", "La respuesta coincide con los documentos"]),
    22: ("chatbot_v22.py",
         ['[paralelo] [usa:delegar {"tarea":"[usa:calcular {\\"expresion\\":\\"2*2\\"}]","modelo":"haiku"}] '
          '[usa:delegar {"tarea":"[usa:calcular {\\"expresion\\":\\"3*3\\"}]","modelo":"haiku"}] '
          '[usa:guardar_nota {"nota":"Prefiere Lisboa"}]'],
         ["[paralelo] 2 herramientas a la vez", "[subagente 1]", "[subagente 2]", "Nota guardada"]),
    23: ("chatbot_v23.py", [TAREA, "tarea", "seguir", "seguir", "s", "seguir"],
         ["Tarea creada con 2 pasos", "Paso 1/2 registrado", "Paso 2/2 registrado",
          "Archivo guardado en salida/guia-para-clientes-nuevos.md", "ya está terminada"]),
}

# Archivos y carpetas que las versiones necesitan para funcionar
NECESARIOS = ["contexto.py", "documentos", "skills", "servidor_mcp_ejemplo.py", "mcp_servidores.json"]


def probar(version: int) -> bool:
    archivo, entradas, esperados, *resto = CASOS[version]
    prohibidos = resto[0] if resto else []
    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = Path(carpeta)
        shutil.copy(PROYECTO / archivo, carpeta)
        for nombre in NECESARIOS:
            origen = PROYECTO / nombre
            if origen.is_dir():
                shutil.copytree(origen, carpeta / nombre)
            elif origen.exists():
                shutil.copy(origen, carpeta)
        proceso = subprocess.run(
            [sys.executable, str(SIMULADOR), archivo],
            input="\n".join(entradas + ["salir"]) + "\n",
            capture_output=True, text=True, cwd=carpeta, timeout=120,
        )
    salida = proceso.stdout + proceso.stderr
    faltan = [e for e in esperados if e not in salida]
    sobran = [e for e in prohibidos if e in salida]
    ok = proceso.returncode == 0 and "Traceback" not in salida and not faltan and not sobran
    print(f"  {'OK  ' if ok else 'FALLA'} v{version:02d} {archivo}")
    if not ok:
        if faltan:
            print(f"        no aparece: {faltan}")
        if sobran:
            print(f"        no debería aparecer: {sobran}")
        print("        " + "\n        ".join(salida.strip().splitlines()[-15:]))
    return ok


def main() -> None:
    versiones = [int(v) for v in sys.argv[1:]] or sorted(CASOS)
    print(f"Probando {len(versiones)} versiones con el Claude simulado (sin API)...")
    resultados = [probar(v) for v in versiones]
    print(f"\n{sum(resultados)}/{len(resultados)} versiones pasaron.")
    sys.exit(0 if all(resultados) else 1)


if __name__ == "__main__":
    main()
