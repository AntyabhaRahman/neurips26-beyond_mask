# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NeurIPS 2026 "Beyond the Mask" work. Python 3.13. The project is in an early scaffold state (only `main.py` and `huggingface-hub` so far), so most architectural decisions are still open — prefer setting up clean foundations over working around legacy structure.

## Toolchain

This project uses **uv** exclusively for environment and dependency management. Do not invoke `pip`, `python -m venv`, or edit `pyproject.toml` dependencies by hand.

```bash
uv sync                          # install/refresh env from uv.lock
uv add <pkg>                     # add a runtime dep (updates pyproject.toml + lock)
uv add --dev <pkg>               # add a dev-only dep
uv remove <pkg>                  # remove a dep
uv run <cmd>                     # run a command inside the project env
uv run python main.py            # run the entrypoint
```

Always prefix Python invocations with `uv run` so the right interpreter and lockfile-resolved deps are used.

## Stack conventions

- **Notebooks: marimo.** Use `.py` marimo notebooks (not Jupyter `.ipynb`). Launch with `uv run marimo edit notebooks/<name>.py` and run headlessly with `uv run marimo run <name>.py`. Marimo notebooks are reactive and stored as plain Python — diffable and importable. Prefer marimo over scratch scripts for any exploratory or analysis work.
  - When unsure about marimo APIs, fetch current docs via the Context7 MCP (library id `marimo-team/marimo`) rather than relying on memory.
- **DataFrames: polars** (not pandas). Use the lazy API (`pl.scan_*` → `.collect()`) for anything non-trivial. Reach for pandas only when interfacing with a library that requires it, and convert at the boundary.
- **Plotting: plotly** (`plotly.express` for quick views, `plotly.graph_objects` for composed figures). Avoid matplotlib/seaborn unless a specific requirement forces it.
- **ML / data: huggingface-hub** is the current dep; expect models/datasets to be loaded from the Hub. Cache to the default HF cache; don't hand-roll downloaders.

## Code style

- Python 3.13, full type hints on every public function and class attribute.
- Google-style docstrings on public functions, classes, and modules. Skip docstrings on trivial private helpers — a clear name is enough.
- Keep functions small and pure where possible; isolate side effects (I/O, model loading) at the edges so the analysis core stays testable in marimo cells.

## Commits

Use Conventional Commit prefixes and keep messages concise (subject line under ~70 chars, imperative mood):

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation only
- `style:` formatting, no logic change
- `refactor:` behavior-preserving code change
- `perf:` performance improvement
- `test:` test-only changes
- `chore:` tooling, deps, build

Example: `feat: add polars loader for masked-image dataset`
