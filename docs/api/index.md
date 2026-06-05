# API reference

The pages below are auto-generated from the package's docstrings via
[mkdocstrings]. Every public symbol — anything listed in a module's
`__all__` or not prefixed with `_` — gets its signature, type hints,
and docstring rendered.

## Public surface

| Subpackage | What's in it |
|---|---|
| [`pid3net.models`](models.md) | Model classes (`PID3Net` and its 2D ablation `PIBaseD3Net`) and the registry (`MODEL_REGISTRY`, `register_model`, `get_spec`, `ModelSpec`). |
| [`pid3net.layers`](layers.md) | Encoders, decoders, conv blocks, activations, physics layers (`RefineLayer`, `CNNTBLayer`), and fusion blocks. |
| [`pid3net.losses`](losses.md) | Diffraction training losses (Poisson NLL, masked MSE) and object-space TV regularisers. |
| [`pid3net.utils`](utils.md) | Dataset loaders and the Keras data generator. |
| [`pid3net.train`](train.md) | Console-script entry point and pipeline helpers. |

## Versioning

The current public API is **v2.0.0**. See the [Changelog](../changelog.md)
for what's new, what's deprecated, and what's removed.

## Conventions

- All docstrings use the **Google style** (`Args:` / `Returns:` /
  `Raises:` sections).
- Tensor shapes are noted with `[B, T, H, W]` notation where `B` is
  batch, `T` is temporal axis, `H × W` is spatial.
- Complex tensors are `complex64` unless noted otherwise.

[mkdocstrings]: https://mkdocstrings.github.io/
