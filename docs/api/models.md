# `pid3net.models`

Model classes, the base class, and the registry used by `pid3net-train`.

::: pid3net.models

## Base class

::: pid3net.models.base_model.PtyBase
    options:
      heading_level: 3
      show_root_heading: true

## Default model

::: pid3net.models.pid3net.PID3Net
    options:
      heading_level: 3
      show_root_heading: true

## 2D ablation

::: pid3net.models.baseline.PIBaseD3Net
    options:
      heading_level: 3
      show_root_heading: true

## Registry

::: pid3net.models.registry
    options:
      heading_level: 3
      show_root_heading: false
      members:
        - ModelSpec
        - MODEL_REGISTRY
        - register_model
        - get_spec
