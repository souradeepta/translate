Run linting and type checking on the bn-en-translate project.

Steps:
1. Activate venv: `source .venv/bin/activate`
2. Run: `make lint` (ruff check)
3. Run: `make typecheck` (mypy strict)
4. Report all issues with file:line references.
5. If there are fixable issues, offer to fix them.
