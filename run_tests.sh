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

PYTEST_ARGS=(-v)

case "${1:-}" in
  --quick)
    # Fast tests only (no API job submissions)
    PYTEST_ARGS+=(-m "not slow")
    ;;
  --slow)
    # Slow tests only, force run regardless of queue status
    PYTEST_ARGS+=(-m slow --run-slow-force)
    ;;
  --all)
    # All tests, force slow tests even if queues busy
    PYTEST_ARGS+=(--run-slow-force)
    ;;
  "")
    # Default: all tests, auto-skip slow if queues busy
    ;;
  *)
    echo "Usage: $0 [--quick|--slow|--all]" >&2
    echo "  (no args)  Run all tests; auto-skip slow tests if queues busy" >&2
    echo "  --quick    Run only fast tests (no API submissions)" >&2
    echo "  --slow     Run only slow tests, ignore queue status" >&2
    echo "  --all      Run all tests, ignore queue status" >&2
    exit 1
    ;;
esac

if [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python -m pytest "${PYTEST_ARGS[@]}"
else
  python3 -m pytest "${PYTEST_ARGS[@]}"
fi
