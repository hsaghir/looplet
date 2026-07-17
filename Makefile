.PHONY: help install style lint format format-check typecheck test check ci clean install-hooks

help:
	@echo "Looplet dev targets:"
	@echo "  make install       - uv sync --all-extras (matches CI)"
	@echo "  make install-hooks - install the pre-push git hook (runs CI checks)"
	@echo "  make style         - repository text style checks"
	@echo "  make lint          - ruff check"
	@echo "  make format        - ruff format (writes)"
	@echo "  make format-check  - ruff format --check (read-only)"
	@echo "  make typecheck     - pyright src/looplet/"
	@echo "  make test          - pytest"
	@echo "  make check         - lint + format-check + typecheck + test (run before commit)"
	@echo "  make ci            - alias for check"

install:
	uv sync --all-extras

style:
	uv run python scripts/check_text_style.py

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright src/looplet/

test:
	uv run pytest --tb=short

# Exactly what CI runs. If `make check` passes, CI passes.
check: install style lint format-check typecheck test
	@echo ""
	@echo "✓ All CI checks passed locally."

ci: check

install-hooks:
	@install -m 0755 scripts/pre-commit.sh .git/hooks/pre-commit
	@install -m 0755 scripts/pre-push.sh .git/hooks/pre-push
	@install -m 0755 .githooks/commit-msg .git/hooks/commit-msg
	@echo "✓ pre-commit hook installed - \`git commit\` auto-formats + lints staged files."
	@echo "✓ commit-msg hook installed - enforces conventional commit format."
	@echo "✓ pre-push hook installed - \`git push\` runs full \`make check\` first."

clean:
	rm -rf .pytest_cache .ruff_cache dist build site coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
