#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
E2E_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
GARDEN_DIR=$(CDPATH= cd -- "$E2E_DIR/.." && pwd)
DATA_DIR="$E2E_DIR/.e2e-data"
DATABASE="$DATA_DIR/stub.sqlite3"
if [ -n "${GARDEN_PYTHON:-}" ]; then
  PYTHON_BIN=$GARDEN_PYTHON
elif [ -x "$GARDEN_DIR/../../.venv/bin/python" ]; then
  PYTHON_BIN="$GARDEN_DIR/../../.venv/bin/python"
else
  PYTHON_BIN=python3
fi

mkdir -p "$DATA_DIR"
rm -f "$DATABASE" "$DATABASE-shm" "$DATABASE-wal"

exec "$PYTHON_BIN" "$GARDEN_DIR/server.py" \
  --mock \
  --host 127.0.0.1 \
  --port 8876 \
  --database "$DATABASE"
