#!/usr/bin/env bash
# One command: start the local service and print one URL.
# Everything runs on this machine. Nothing is uploaded anywhere.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PORT="${SPATIAL_AI_PORT:-8420}"

if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null; then
  PYTHON="python3"
else
  echo "python3 is required"
  exit 1
fi
"$PYTHON" - <<'PY' || { echo "Missing dependencies. Run: .venv/bin/pip install -r pipeline/requirements.txt fastapi uvicorn python-multipart"; exit 1; }
import importlib.util, sys
missing = [m for m in ("numpy", "PIL", "jsonschema", "fastapi", "uvicorn", "multipart")
           if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

echo "Spatial Review — http://127.0.0.1:${PORT}"
echo "Local only. Captures and generated artifacts never leave this machine."
exec "$PYTHON" -m uvicorn service.api:app --host 127.0.0.1 --port "${PORT}"
