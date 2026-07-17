#!/usr/bin/env bash
# Pre-commit hook - auto-formats staged files and blocks the commit on
# lint errors.  Fast (< 1s on a typical change).  Pairs with the
# pre-push hook which runs the full CI check.
#
# To install:
#   make install-hooks
#
# To bypass for a single commit (not recommended):
#   git commit --no-verify
set -euo pipefail

# Only operate on staged Python files; skip when none are staged.
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)
if [[ -z "$STAGED_PY" ]]; then
    exit 0
fi

echo "→ pre-commit: checking staged Python files"

# 1. Auto-format.  If ruff rewrites anything, re-stage it so the commit
#    includes the formatted version.
if ! uv run ruff format --check $STAGED_PY >/dev/null 2>&1; then
    echo "  formatting..."
    uv run ruff format $STAGED_PY
    git add $STAGED_PY
    echo "  ✓ reformatted and re-staged"
fi

# 2. Lint.  Fail fast on errors - developer must fix before committing.
if ! uv run ruff check $STAGED_PY; then
    echo ""
    echo "✗ pre-commit: lint errors above.  Fix them or run"
    echo "    uv run ruff check --fix $STAGED_PY"
    echo "  then \`git add\` and retry."
    exit 1
fi

echo "  ✓ format + lint clean"
