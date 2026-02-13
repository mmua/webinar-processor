#!/usr/bin/env python3
import json
import subprocess
import sys

def run_ruff_json() -> list[dict]:
    # Ruff supports JSON output for machine parsing.
    # If Ruff exits non-zero due to lint findings, we still want the output.
    proc = subprocess.run(
        ["ruff", "check", ".", "--output-format", "json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Ruff may print non-JSON noise to stderr; JSON is on stdout for --output-format json.
    out = proc.stdout.strip()
    if not out:
        # If nothing returned, treat as zero issues (or an error case).
        if proc.returncode not in (0, 1):
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)
        return []

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        print("Failed to parse Ruff JSON output.", file=sys.stderr)
        print("STDOUT:", proc.stdout, file=sys.stderr)
        print("STDERR:", proc.stderr, file=sys.stderr)
        raise

def main() -> None:
    issues = run_ruff_json()
    # Each JSON entry represents one finding.
    total = len(issues)

    result = {
        "tool": "ruff",
        "command": "ruff check . --output-format json",
        "total_violations": total,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
