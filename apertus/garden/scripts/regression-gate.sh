#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
garden_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
e2e_dir="$garden_dir/e2e"
mode=${1:-gate}

if [ -n "${GARDEN_PYTHON:-}" ]; then
  python_bin=$GARDEN_PYTHON
elif [ -x "$garden_dir/../../.venv/bin/python" ]; then
  python_bin="$garden_dir/../../.venv/bin/python"
else
  python_bin=python3
fi

run_contracts() {
  "$python_bin" -m pytest \
    -c "$garden_dir/../../pyproject.toml" \
    -q \
    -m contract \
    "$garden_dir/tests"
}

run_python_suite() {
  "$python_bin" -m pytest \
    -c "$garden_dir/../../pyproject.toml" \
    -q \
    "$garden_dir/tests"
}

run_performance_budget() {
  "$python_bin" "$garden_dir/scripts/performance-budget.py"
}

run_playwright() {
  suite=$1
  if [ ! -d "$e2e_dir/node_modules/@playwright/test" ]; then
    echo "Playwright není nainstalovaný. Spusť: cd $e2e_dir && npm install" >&2
    exit 2
  fi
  (
    cd "$e2e_dir"
    npm run "test:$suite"
  )
}

case "$mode" in
  contracts)
    run_contracts
    ;;
  performance)
    run_performance_budget
    ;;
  gate)
    run_python_suite
    run_performance_budget
    run_playwright stub
    ;;
  live)
    run_python_suite
    run_performance_budget
    run_playwright stub
    run_playwright live
    ;;
  *)
    echo "Použití: $0 [contracts|performance|gate|live]" >&2
    exit 2
    ;;
esac
