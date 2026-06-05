---
title: PID3Net
hide:
  - navigation
---

# PID3Net

**Physics-Informed Deep learning Network for Dynamic Diffraction imaging.**

PID3Net is a self-supervised, physics-informed network for time-resolved
coherent X-ray diffraction imaging (CXDI). It combines a 3D temporal
encoder–decoder with an iterative, differentiable refinement block that
mirrors classical phase-retrieval algorithms (ePIE, RAAR) — so the
network's outputs stay consistent with the underlying diffraction
physics rather than relying solely on learned statistical patterns.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Read the design log](concepts/architecture.md){ .md-button }
[API reference](api/index.md){ .md-button }

---

## At a glance

=== "Install"

    ```bash
    pip install -e .
    ```

=== "Train"

    ```bash
    pid3net-train configs/Moving_chart_1ms.yaml
    ```

=== "Import as a library"

    ```python
    from pid3net.models import PID3Net, MODEL_REGISTRY
    from pid3net.layers.physics_layers import RefineLayer
    ```

## What's inside

- **Two model variants** — the default 3D temporal `PID3Net` and its
  2D ablation `PIBaseD3Net`, dispatched through a single registry. See
  [Architecture](concepts/architecture.md).
- **Physics-informed refinement** — `RefineLayer` runs `n_step`
  iterations of ePIE or RAAR with a learned CNN correction per step,
  in either *polar* (`amp · exp(jφ)`) or *refractive* (`φ + j·amp`)
  reconstruction mode. See [Physics](concepts/physics.md).
- **YAML-driven training** — single config file controls the entire
  pipeline; CLI flags override individual keys. See
  [Configuration](guides/configuration.md).
- **Pip-installable** — modern PEP 621 `pyproject.toml`,
  `pid3net-train` console script, optional `[dev]` and `[docs]` extras.

## Project status

Current release: **v2.0.0** — see the [Changelog](changelog.md) for the
full list of additions, breaking renames (`PID3NetV3` → `PID3Net`), and
removals.
