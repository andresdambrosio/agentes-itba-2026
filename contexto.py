"""Monitor de contexto para las variantes con conteo de las v01, v02 y v03. Solo mide; no guarda conversación.

Usa los tokens reportados por la API, sin llamadas adicionales.
"""


class MonitorContexto:
    def __init__(self, ventana: int):
        self.ventana = ventana
        self.turnos = []

    def mostrar(self, usage, mensajes: int):
        # Los tokens de caché también ocupan contexto, cuando los hay.
        entrada = (usage.input_tokens
                   + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                   + (getattr(usage, "cache_creation_input_tokens", 0) or 0))
        salida = usage.output_tokens
        self.turnos.append((mensajes, entrada, salida))

        print(f"[contexto] Ventana: {self.ventana:,} tokens")
        print(" Llamada | Mensajes | Entrada | Salida | Entrada + salida | % ventana")
        print("---------+----------+---------+--------+------------------+----------")
        for numero, (cantidad, tokens_in, tokens_out) in enumerate(self.turnos, 1):
            total = tokens_in + tokens_out
            print(f"{numero:8} | {cantidad:8} | {tokens_in:7,} | {tokens_out:6,} | "
                  f"{total:16,} | {total / self.ventana:8.4%}")

        total = entrada + salida
        llenos = min(30, int(total / self.ventana * 30))
        barra = "#" * llenos + "-" * (30 - llenos)
        print(f" Última llamada: [{barra}] {total / self.ventana:.4%}")
        print(" Entrada: todo lo enviado. Salida: tokens generados, incluido razonamiento.")
        print(" Cada fila mide una llamada; los turnos no se suman para ocupar la ventana.\n")
