#!/usr/bin/env bash
# CONSILIUM vision server (Linux/macOS). First run: creates .venv, installs
# GPU PyTorch + packages, downloads the models.
set -euo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
if [ ! -x .venv/bin/python ]; then
  echo "[setup] Creating Python environment in .venv ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  echo "[setup] Installing PyTorch with CUDA from $TORCH_INDEX ..."
  .venv/bin/python -m pip install torch --index-url "$TORCH_INDEX"
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python server.py
