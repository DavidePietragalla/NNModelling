# NNModelling Documentation

Sphinx documentation for the NNModelling project.

## Prerequisites

- Python 3.10+
- `uv` (Python package manager)

## Build

```bash
cd docs2
uv run make html
```

HTML output is generated in `build/html/`.

To clean build artifacts:

```bash
uv run make clean
```

## Structure

- `source/` — RST source files
- `build/` — compiled output (git-ignored)
- `Makefile` — Sphinx targets (html, clean, etc.)
- `requirements.txt` — Python dependencies
