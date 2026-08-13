# Contributing Guide

Thanks for contributing to this project.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r web/backend/requirements.txt
```

## Code Style

- Keep code modular inside `src/asag/`.
- Prefer small, testable functions.
- Use clear names for scripts under `scripts/`.
- Avoid committing generated artifacts and local environment files.

## Typical Workflow

1. Create a branch from `main`.
2. Make focused changes.
3. Run local checks:

```bash
PYTHONPYCACHEPREFIX='./tmp/pycache' ./.venv/bin/python -m py_compile src/asag/*.py scripts/*.py web/backend/app/main.py streamlit_app.py
```

4. Run a smoke test:

```bash
python3 scripts/run_best_pipeline.py --skip-cross-encoder
```

5. Commit with a clear message.
6. Open a pull request with:
- what changed
- why it changed
- how it was tested

## Pull Request Checklist

- [ ] Code compiles.
- [ ] README/commands updated if behavior changed.
- [ ] No sensitive files added.
- [ ] No local environment files committed.
- [ ] `.gitignore` rules respected.

## Reporting Issues

When filing an issue, include:
- exact command used
- full error traceback
- OS and Python version
- whether `.venv` was active
