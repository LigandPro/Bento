# Contributing

## Development Workflow

1. Create a feature branch from `main`.
2. Install dependencies with uv:
   - `uv sync --no-editable --extra lint --extra test`
3. Run checks locally:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run pytest`
4. Open a pull request with a clear description of the change and validation steps.

## Coding Standards

- Use Python 3.10+ compatible syntax.
- Keep scripts and docs in English.
- Prefer small, reviewable commits.
- Include tests for behavior changes.

## Data and Large Files

- Do not commit generated artifacts or local runtime caches.
- Keep large benchmark artifacts outside the repository when possible and document retrieval steps in PRs.
