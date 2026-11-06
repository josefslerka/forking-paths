#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../../../.." && pwd)
if [ -n "${GARDEN_PYTHON:-}" ]; then
  python_bin=$GARDEN_PYTHON
elif [ -x "$repo_dir/.venv/bin/python" ]; then
  python_bin="$repo_dir/.venv/bin/python"
else
  python_bin=python3
fi
exec "$python_bin" "$script_dir/start-model-admin.py" "$@"
