#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf test-runs

if [[ ! -f ".env" ]]; then
  echo "Missing .env file in $SCRIPT_DIR" >&2
  exit 1
fi

set -a
source .env
set +a

if [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python -m pytest
else
  python3 -m pytest
fi
