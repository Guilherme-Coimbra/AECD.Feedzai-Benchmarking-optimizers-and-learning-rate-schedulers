# Source Modules Directory

All reusable Python code lives in `src/`. Organize shared utilities, data loaders, model definitions, and training routines here so notebooks and experiments can import them consistently.

## Structure Guidelines

- Group functionality into focused modules (`data_loading.py`, `preprocessing.py`, `modeling.py`, `utils.py`, etc.).
- Prefer packages (folders with `__init__.py`) when modules grow large or require nested organization.
- Keep side effects out of module import time; expose functions, classes, and configuration constants only.

## Development Practices

- Use type hints and docstrings to clarify interfaces.
- Add unit tests under a top-level `tests/` directory when feasible.
- Centralize configuration defaults here and allow overrides via experiment configs.
- Avoid hard-coding paths; accept them as parameters or load from environment variables.
- After creating or updating modules, install the project in editable mode (`pip install -e .`) so any environment can import them without manual `sys.path` tweaks.
- Import modules via the package namespace, e.g., `from src import utils` or `from src.io_ops import load_dataset`.

Record additional conventions or architectural decisions in this README as the codebase evolves.
