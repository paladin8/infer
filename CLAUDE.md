# infer

Educational LLM inference runtime. See `docs/OVERALL_DESIGN.md` for the full design plan.

## Commands

- `uv run pytest` — run tests
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run mypy .` — type check

## Conventions

- Python 3.14+, target set in `.python-version`
- Use `uv sync --dev` to set up the environment
- Ruff handles linting and formatting (line length 100, double quotes, space indent)
- All code must pass ruff and mypy before commit (pre-commit hooks enforce this)
- Tests go in `tests/` and are run with pytest
- Use type annotations on all function signatures
- Always place imports at the top of the file, not inline
