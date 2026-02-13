#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

BASELINE_FILE=".opencode/ruff-baseline.json"
SCORE_SCRIPT=".opencode/skills/python-ruff-guard/scripts/ruff_score.py"

ensure_ruff() {
  if ! command -v ruff >/dev/null 2>&1; then
    echo "ERROR: 'ruff' not found on PATH. Install Ruff in your environment."
    exit 2
  fi
}

mkdir -p .opencode

ensure_ruff

case "$MODE" in
  baseline)
    echo "==> Creating Ruff baseline..."
    python3 "$SCORE_SCRIPT" > "$BASELINE_FILE"
    echo "Baseline written to $BASELINE_FILE"
    cat "$BASELINE_FILE"
    ;;
  gate)
    if [[ ! -f "$BASELINE_FILE" ]]; then
      echo "ERROR: Baseline file not found at $BASELINE_FILE"
      echo "Run: bash .opencode/skills/python-ruff-guard/scripts/ruff_guard.sh baseline"
      exit 2
    fi

    echo "==> Computing current Ruff score..."
    CURRENT_JSON="$(python3 "$SCORE_SCRIPT")"

    BASELINE_COUNT="$(python3 -c 'import json; print(json.load(open("'"$BASELINE_FILE"'"))["total_violations"])')"
    CURRENT_COUNT="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["total_violations"])' "$CURRENT_JSON")"

    echo "Baseline violations: $BASELINE_COUNT"
    echo "Current violations:   $CURRENT_COUNT"

    if [[ "$CURRENT_COUNT" -gt "$BASELINE_COUNT" ]]; then
      echo "FAIL: Ruff violations increased (quality decreased)."
      echo "Suggested fixes:"
      echo "  - ruff check . --fix"
      echo "  - (optional) ruff format ."
      exit 1
    fi

    echo "PASS: Ruff violations did not increase."
    ;;
  *)
    echo "Usage:"
    echo "  $0 baseline   # store baseline before changes"
    echo "  $0 gate       # compare after changes; fail if worse"
    exit 2
    ;;
esac
