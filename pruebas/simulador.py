"""Ejecuta un chatbot_vN.py con un Claude falso: sin red, sin clave y sin costo.

Sirve para comprobar que el código de cada versión funciona (el bucle, las
herramientas, la memoria, los errores), no la calidad de las respuestas.
Reemplaza anthropic.Anthropic por un cliente falso que devuelve objetos
reales del SDK y revisa que cada petición sea válida (por ejemplo, que todo
tool_use tenga su tool_result).

Uso: python pruebas/simulador.py chatbot_vN.py < entradas.txt
(normalmente lo usa pruebas/probar_todas.py)

El "Claude" falso obedece marcas dentro del texto del usuario:
  [usa:herramienta {json}]  -> responde con un tool_use de esa herramienta (si hay
                               varias marcas, usa la última)
  [paralelo]                -> con varias marcas [usa:...], las pide todas juntas
  [falla]                   -> la llamada lanza APIConnectionError
  [falla-despues]           -> falla la llamada que sigue a un tool_result
                               (para probar qué pasa si el turno se corta
                               después de ejecutar una acción)
  [corta]                   -> stop_reason max_tokens
  [rechaza]                 -> stop_reason refusal
Después de un tool_result responde con texto que incluye el resultado.
Con output_config.format devuelve JSON con el primer valor del enum (o true
para los booleanos). Con [verifica-mal] en la conversación, el revisor de la
v21 contesta que la respuesta tiene un problema.
"""
import json
import re
import runpy
import sys
import itertools
from pathlib import Path

import anthropic

try:
    import httpx2 as httpx  # anthropic 2.x
except ImportError:
    import httpx  # versiones anteriores del SDK

_ids = itertools.count(1)
LOG = []


def normalizar(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json", exclude_none=True)
    if isinstance(x, list):
        return [normalizar(i) for i in x]
    if isinstance(x, dict):
        return {k: normalizar(v) for k, v in x.items()}
    return x


def validar(kw):
    kw["messages"] = normalizar(kw["messages"])
    msgs = kw["messages"]
    assert msgs and msgs[0]["role"] == "user", "el primer mensaje debe ser del usuario"
    for i, m in enumerate(msgs):
        c = m["content"]
        if isinstance(c, list):
            for b in c:
                assert isinstance(b, dict), f"bloque no serializable en msg {i}: {type(b)}"
                if b["type"] == "tool_use":
                    sig = msgs[i + 1] if i + 1 < len(msgs) else None
                    assert sig and sig["role"] == "user" and isinstance(sig["content"], list), \
                        f"tool_use sin tool_result en msg {i}"
                    ids = {x.get("tool_use_id") for x in sig["content"]}
                    assert b["id"] in ids, f"tool_use {b['id']} sin resultado"
    json.dumps(kw["messages"])  # debe ser serializable
    json.dumps(kw.get("system", ""))
    for t in kw.get("tools", []):
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", t["name"]), t["name"]


def marcas(texto):
    """Encuentra las marcas [usa:nombre {json}] del texto, sin contar las que
    están dentro del JSON de otra (como la tarea de un delegar)."""
    decodificador, i, encontradas = json.JSONDecoder(), 0, []
    while (k := texto.find("[usa:", i)) != -1:
        m = re.match(r"\[usa:(\w+) ", texto[k:])
        if not m:
            i = k + 1
            continue
        try:
            argumentos, fin = decodificador.raw_decode(texto, k + m.end())
        except ValueError:
            i = k + 1
            continue
        encontradas.append((m.group(1), argumentos))
        i = fin
    return encontradas


def ultimo_texto_usuario(msgs):
    for m in reversed(msgs):
        if m["role"] == "user" and isinstance(m["content"], str):
            return m["content"]
    return ""


def responder(kw):
    validar(kw)
    LOG.append(kw)
    msgs = kw["messages"]
    ult = msgs[-1]
    fmt = (kw.get("output_config") or {}).get("format")
    content, stop = [], "end_turn"
    if fmt:
        props = fmt["schema"]["properties"]
        out = {k: True if v.get("type") == "boolean" else (v.get("enum") or ["x"])[0]
               for k, v in props.items()}
        if "correcta" in props and "[verifica-mal]" in json.dumps(msgs, ensure_ascii=False):
            out = {"correcta": False, "problemas": "Dice 30% y el manual dice 15%."}
        content = [{"type": "text", "text": json.dumps(out)}]
    elif isinstance(ult["content"], list) and any(b.get("type") == "tool_result" for b in ult["content"]):
        if "[falla-despues]" in ultimo_texto_usuario(msgs):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        res = " | ".join(str(b.get("content"))[:80] for b in ult["content"] if b.get("type") == "tool_result")
        content = [{"type": "text", "text": f"Listo. Resultado: {res}"}]
    else:
        texto = ult["content"] if isinstance(ult["content"], str) else ultimo_texto_usuario(msgs)
        if "[falla]" in texto:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        pedidas = marcas(texto)
        if pedidas and "[paralelo]" not in texto:
            pedidas = pedidas[-1:]
        tool_choice = (kw.get("tool_choice") or {}).get("type")
        if pedidas and kw.get("tools") and tool_choice != "none" and isinstance(ult["content"], str):
            content = [
                {"type": "thinking", "thinking": "", "signature": "firma-falsa"},
                {"type": "text", "text": "Voy a usar una herramienta."},
            ] + [{"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": nombre, "input": argumentos}
                 for nombre, argumentos in pedidas]
            stop = "tool_use"
        elif "[corta]" in texto:
            content = [{"type": "text", "text": "Respuesta cort"}]
            stop = "max_tokens"
        elif "[rechaza]" in texto:
            content, stop = [], "refusal"
        else:
            # Como la API real: el razonamiento ("thinking") llega antes del texto
            content = [{"type": "thinking", "thinking": "", "signature": "firma-falsa"},
                       {"type": "text", "text": f"Respuesta simulada a: {texto[:60]} (modelo {kw['model']})"}]
    tam = len(json.dumps(msgs)) // 4
    return anthropic.types.Message.model_validate({
        "id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "model": kw["model"],
        "content": content, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": tam, "output_tokens": 20,
                  "cache_read_input_tokens": tam // 2 if len(msgs) > 1 else 0,
                  "cache_creation_input_tokens": 0},
    })


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        for b in self.msg.content:
            if b.type == "text":
                yield from (b.text[i:i + 7] for i in range(0, len(b.text), 7))

    def get_final_message(self):
        return self.msg


class _Messages:
    def create(self, **kw):
        return responder(kw)

    def stream(self, **kw):
        return _Stream(responder(kw))


class FakeAnthropic:
    def __init__(self, *a, **k):
        self.messages = _Messages()


anthropic.Anthropic = FakeAnthropic

if __name__ == "__main__":
    ruta = sys.argv[1]
    sys.argv = [ruta]
    # Igual que al ejecutar el script directamente: sus módulos vecinos son importables.
    sys.path.insert(0, str(Path(ruta).resolve().parent))
    runpy.run_path(ruta, run_name="__main__")
    print(f"\n[simulador] llamadas a la API: {len(LOG)}", file=sys.stderr)
