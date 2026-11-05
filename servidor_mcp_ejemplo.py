"""Servidor MCP de ejemplo para la v19.

MCP (Model Context Protocol) es un estándar para ofrecer herramientas a
cualquier agente. Un servidor MCP es un programa aparte que dice "tengo
estas herramientas" y las ejecuta cuando un cliente se lo pide. El agente
(cliente) no necesita saber cómo están programadas.

Este servidor ofrece tres herramientas útiles para Nimbo Viajes:
    - dias_hasta(fecha): cuántos días faltan para una fecha (sirve para la
      política de cancelación).
    - convertir_moneda(monto, de, a): conversión con tasas FICTICIAS.
    - guardar_recordatorio(fecha, texto): anota un recordatorio en
      recordatorios.txt.

Las dos primeras se declaran de solo lectura (read_only_hint=True): no
cambian nada, así que la v19 las ejecuta sin preguntar. La tercera escribe
un archivo y no se declara así: la v19 le pide permiso al usuario antes.

No hace falta ejecutarlo a mano: chatbot_v19.py lo lanza solo (ver
mcp_servidores.json). Si querés probarlo por tu cuenta:
    pip install mcp
    python servidor_mcp_ejemplo.py     # queda esperando mensajes por stdin
"""

from datetime import date
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

servidor = MCPServer("utilidades-viaje")

# Tasas inventadas para el ejemplo: cuántos dólares vale 1 unidad de cada moneda
TASAS_EN_USD = {"USD": 1.0, "EUR": 1.08, "ARS": 0.001, "COP": 0.00025, "UYU": 0.025}


# Anotación: esta herramienta solo lee, no modifica nada
SOLO_LECTURA = ToolAnnotations(read_only_hint=True)


@servidor.tool(annotations=SOLO_LECTURA)
def dias_hasta(fecha: str) -> str:
    """Cuenta cuántos días faltan desde hoy hasta una fecha (formato AAAA-MM-DD).
    Si la fecha ya pasó, el resultado es negativo."""
    try:
        objetivo = date.fromisoformat(fecha)
    except ValueError:
        raise ToolError("La fecha debe tener el formato AAAA-MM-DD, por ejemplo 2026-12-25")
    dias = (objetivo - date.today()).days
    return f"Faltan {dias} días para el {objetivo.isoformat()} (hoy es {date.today().isoformat()})."


@servidor.tool(annotations=SOLO_LECTURA)
def convertir_moneda(monto: float, de: str, a: str) -> str:
    """Convierte un monto entre USD, EUR, ARS, COP y UYU. Las tasas son
    FICTICIAS, solo para el ejemplo."""
    de, a = de.upper(), a.upper()
    if de not in TASAS_EN_USD or a not in TASAS_EN_USD:
        # ToolError: un error "esperado" cuyo mensaje le llega al modelo
        raise ToolError(f"Monedas disponibles: {', '.join(TASAS_EN_USD)}")
    resultado = monto * TASAS_EN_USD[de] / TASAS_EN_USD[a]
    return f"{monto:,.2f} {de} = {resultado:,.2f} {a} (tasa ficticia de ejemplo)"


# Sin anotación de solo lectura: escribe en disco, así que el cliente debería
# pedir permiso antes de usarla.
@servidor.tool()
def guardar_recordatorio(fecha: str, texto: str) -> str:
    """Guarda un recordatorio para una fecha (formato AAAA-MM-DD) en el
    archivo recordatorios.txt."""
    try:
        date.fromisoformat(fecha)
    except ValueError:
        raise ToolError("La fecha debe tener el formato AAAA-MM-DD, por ejemplo 2026-12-25")
    with open(Path(__file__).parent / "recordatorios.txt", "a", encoding="utf-8") as archivo:
        archivo.write(f"{fecha}: {texto}\n")
    return f"Recordatorio guardado para el {fecha}."


if __name__ == "__main__":
    servidor.run()  # por defecto se comunica por stdin/stdout ("stdio")
