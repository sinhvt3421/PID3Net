# Changelog

All notable changes to **PID3Net** are documented in this file.

The format is based on [Keep a Changelog][kac] and this project
adheres to [Semantic Versioning][semver].

## [2.0.0] — 2026-06-05

### Removed
- External baseline models `AutoPhaseNN` and `PtychoNN` (registry keys
  `autonn` and `ptychonn`) and their adapted source files. The project
  now ships only the `PID3Net` 3D temporal model (`3d3`) and its 2D
  ablation `PIBaseD3Net` (`2d`). Existing trained checkpoints under
  `--model 3d3` or `--model 2d` continue to load unchanged; runs that
  used `--model autonn` or `--model ptychonn` will need to switch
  models or pin to v1.0.0.

### Added

- **Noise-aware modulus projection** in `pid3net.layers.physics_layers.RefineLayer`
  with new opt-in flags under a new `refine:` YAML block:
    - `refine.poisson_projection` — replaces the hard Gaussian-MLE
      modulus projection with one Poisson-MLE gradient step
      `Ψ ← Ψ − η·Ψ·(1 − I_target / (|Ψ|² + ε))` per refinement
      iteration. Mutually exclusive with the Gaussian projection.
  Both default off — existing configs are unaffected.

- **`pid3net.models.MODEL_REGISTRY`** — a registry-based dispatch for
  model variants. Adding a model now takes one file plus one registry
  entry; no edits to `base_model.py` or `train.py`.

- **`pid3net-train` console-script entry point**, declared in
  `pyproject.toml`'s `[project.scripts]`. Equivalent invocations:
  ```bash
  pid3net-train configs/Moving_chart_1ms.yaml      # installed
  python -m pid3net.train configs/Moving_chart_1ms.yaml
  python train_ssp.py        configs/Moving_chart_1ms.yaml   # legacy shim
  ```

- **Documentation site** built with MkDocs Material + mkdocstrings,
  auto-deployed to GitHub Pages via `.github/workflows/docs.yml`.

### Changed
- README CLI table fixed: argument was documented as `--mode` but the
  parser uses `--model`. Added missing `--rec_mode` and
  `--update_method` entries.

- `requirements.txt` aligned with `pyproject.toml`; the latter is now
  the source of truth and the former is a reproducibility lock.
  
- `RefineLayer.apply_intensity_constraint` split into clean
  `_gaussian_projection` and `_poisson_projection` methods; the
  Gaussian path is bit-identical to the pre-change behaviour when all
  `refine.*` flags default off.


## [1.0.0] — 2025-03-XX

Initial public release: 3D temporal encoder-decoder model with
physics-informed iterative refinement (PIE), refractive and
polar reconstruction modes, four probe modes (single, single_c,
multi, multi_c), Poisson and MSE diffraction losses, and YAML-driven training pipeline.

[2.0.0]: https://github.com/sinhvt3421/PID3Net/releases/tag/v2.0.0
[1.0.0]: https://github.com/sinhvt3421/PID3Net/releases/tag/v1.0.0
