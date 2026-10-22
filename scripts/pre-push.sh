#!/usr/bin/env bash
# Pre-push hook - runs the same checks as CI so you never push a
# broken commit. To install:
#
#   cp scripts/pre-push.sh .git/hooks/pre-push
#   chmod +x .git/hooks/pre-push
#
# Or simply: make install-hooks
set -euo pipefail

git_local_variables=$(git rev-parse --local-env-vars)
while IFS= read -r variable; do
	unset "$variable"
done <<< "$git_local_variables"

echo "→ Running CI checks locally before push..."
make check
