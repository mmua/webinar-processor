# Skill: python-ruff-guard

Purpose:
- Enforce Ruff lint quality gates before and after code changes.
- Ensure code quality does NOT decrease: total Ruff violations must not increase compared to baseline.

When to use:
- Before starting any implementation or refactor that touches Python code.
- After completing changes (before considering the task "done").

Assumptions:
- Ruff is available (prefer `ruff` on PATH).

Workflow (MANDATORY):
1) Establish a baseline BEFORE making changes:
   - Run: `bash .opencode/skills/python-ruff-guard/scripts/ruff_guard.sh baseline`
   - This stores a baseline score file under `.opencode/ruff-baseline.json`.

2) After changes, run the gate:
   - Run: `bash .opencode/skills/python-ruff-guard/scripts/ruff_guard.sh gate`
   - This compares current Ruff violations vs the baseline.
   - If current > baseline, code quality decreased → MUST fix issues (preferred) or revert.
   - If current <= baseline, quality did not decrease → OK.

Fix strategy (preferred order):
- Run `ruff check . --fix` to auto-fix issues that Ruff can fix.
- Optionally run `ruff format .` if the project uses Ruff formatting.
- Re-run the gate until it passes.

Notes:
- Use `ruff check .` as the standard lint command.
- Treat a lower or equal violation count as "pass". Higher count is "fail".
