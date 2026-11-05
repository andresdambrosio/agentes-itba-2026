#!/bin/bash
# Corre el chatbot con el Python de .venv y la clave de Anthropic.
#
#   ./correr.sh                       la última versión
#   ./correr.sh 15                    chatbot_v15.py (5 o 05: chatbot_v05.py)
#   ./correr.sh evaluar.py --busqueda cualquier .py, con sus argumentos

set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No encuentro .venv. Crealo con Python 3.10 o más nuevo e instalá requirements.txt." >&2
    exit 1
fi

# La clave se lee de .env; si no está, se pide una sola vez y queda guardada ahí.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
    read -r -s -p "Pegá tu ANTHROPIC_API_KEY (no se ve al escribir): " ANTHROPIC_API_KEY
    echo
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "No ingresaste ninguna clave." >&2
        exit 1
    fi
    export ANTHROPIC_API_KEY
    touch .env
    chmod 600 .env
    echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> .env
    echo "Clave guardada en .env"
fi

case "$1" in
    "")
        ultima=$(ls chatbot_v*.py | sed -E 's/chatbot_v([0-9]+)\.py/\1/' | sort -n | tail -1)
        archivo="chatbot_v$ultima.py"
        ;;
    *[!0-9]*)
        archivo="$1"
        shift
        ;;
    *)
        # Los archivos van con dos dígitos (chatbot_v05.py); 10# evita que 08 se lea como octal
        archivo=$(printf "chatbot_v%02d.py" "$((10#$1))")
        shift
        ;;
esac

if [ ! -f "$archivo" ]; then
    echo "No existe $archivo" >&2
    exit 1
fi

echo "Corriendo $archivo"
exec .venv/bin/python "$archivo" "$@"
