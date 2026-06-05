# Contributing

Thanks for your interest in improving PID3Net.

## Development environment

```bash
git clone https://github.com/sinhvt3421/PID3Net
cd PID3Net
pip install -e ".[dev,docs]"
```

This installs the runtime dependencies, plus `pytest`, `ruff`, `black`,
and the docs tooling.

## Code style

- **Format**: `black .`
- **Lint**: `ruff check .`
- **Type hints**: encouraged for new public APIs; not retro-enforced.
- **Docstrings**: Google style, with explicit `Args` / `Returns` /
  `Raises` sections. Tensor shapes documented as `[B, T, H, W]`.

## Running the docs locally

```bash
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # CI parity — fails on any rendering warning
```

The docs build does not require TensorFlow — `mkdocstrings` parses the
source directly. So the dev loop for documentation changes is fast
even on machines without a GPU.

## Tests

A `tests/` directory and CI workflow are planned (see
[Changelog](changelog.md)). Until they ship, contributors should at
minimum:

- `python -m py_compile` any touched `.py` file.
- Run a single epoch on `configs/Moving_chart_1ms.yaml` and confirm
  loss is finite and the inference output writes.

## Release checklist

When cutting a new release:

1. Update `pid3net/__init__.py` → `__version__`.
2. Add a new `## [X.Y.Z] — YYYY-MM-DD` section to `CHANGELOG.md`
   above `## [Unreleased]`. Move any pending entries down into the
   new section.
3. Add the version link at the bottom:
   `[X.Y.Z]: https://github.com/sinhvt3421/PID3Net/releases/tag/vX.Y.Z`
4. Update the `Unreleased` link's compare base to `vX.Y.Z`.
5. Commit, tag `vX.Y.Z`, push. The `Deploy docs` workflow rebuilds
   the site on push to `main`.
6. (Optional) `python -m build && twine upload dist/*` if publishing
   to PyPI.

Keep `pid3net.__version__` and the top `## [X.Y.Z]` in `CHANGELOG.md`
**in sync**. The docs site surfaces both.

## Reporting issues

Open an issue on the GitHub repo with:

- The exact command that failed (and the YAML config).
- The full traceback.
- The output of `python -c "import pid3net; print(pid3net.__version__)"`
  and `pip show pid3net`.

## Pull requests

Small, focused PRs are easier to review. If you're touching the
refinement layer or the loss functions, please:

1. Verify numerical regression on `configs/Moving_chart_1ms.yaml` —
   loss curve should match within RNG noise when your change is
   disabled.
2. Add a `## [Unreleased]` entry to `CHANGELOG.md`.
3. Update the relevant concept / guide page in `docs/`.
4. Re-run `mkdocs build --strict` to catch broken cross-links.
