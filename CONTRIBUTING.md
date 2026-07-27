# Contributing

Focused bug fixes, compatibility updates, and reproducible evaluations are welcome.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test,dev]"
pytest
ruff check src tests examples
ruff format --check src tests examples
python -m build
twine check dist/*
```

Install `.[train]` only when working on the GPU training path.

## Pull requests

- Keep changes scoped and explain the behavior they alter.
- Add a regression test for fixes and a reproducible measurement for empirical claims.
- Do not modify `competition/training.py`; it is the archived competition submission.
- Do not commit competition data, model weights, adapters, tokens, or generated training
  runs.
- Run the light test suite and package checks before opening a pull request.

For benchmark changes, include the seed, data split, model revision, dependency versions,
hardware, command, and raw machine-readable metrics.
