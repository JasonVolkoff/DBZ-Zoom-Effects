#!/bin/bash
# One-time setup: creates a virtual environment and installs dependencies.
set -e
cd "$(dirname "$0")"

PY=""
for v in python3.12 python3.11 python3.10 python3; do
  if command -v "$v" >/dev/null 2>&1; then PY="$v"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 not found. Install it with:  brew install python@3.12"
  exit 1
fi
echo "Using $($PY --version)"

$PY -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo
echo "Done. Start it with:  ./run.sh"
