#!/usr/bin/env bash
# ============================================================================
# MamNexa AI - Phase I environment bootstrap (Linux / macOS / Colab shell)
# Creates a local virtual environment and installs pinned dependencies.
#
# PREREQUISITE: a real Python 3.11/3.12 interpreter on PATH.
#   Verify with:  python3 --version
#
# On Google Colab you typically skip the venv and just run:
#   !pip install -r requirements.txt
# ============================================================================
set -euo pipefail

# Move to the project root (parent of this scripts/ dir) regardless of CWD.
cd "$(dirname "$0")/.."

echo "[1/4] Verifying Python interpreter..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "  ERROR: 'python3' not found. Install CPython 3.11/3.12 first." >&2
    exit 1
fi
python3 --version

echo "[2/4] Creating virtual environment at .venv ..."
python3 -m venv .venv

echo "[3/4] Upgrading pip ..."
./.venv/bin/python -m pip install --upgrade pip

echo "[4/4] Installing pinned requirements ..."
./.venv/bin/python -m pip install -r requirements.txt

cat <<'EOF'

Done. Activate the environment with:
    source .venv/bin/activate
Then run the Phase I verification tests with:
    python -m pytest tests -v
EOF
