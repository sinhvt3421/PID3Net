# Quickstart

After [installing PID3Net](installation.md), train the default `PID3Net`
model on the bundled moving-chart config.

## Train

```bash
pid3net-train configs/Moving_chart_1ms.yaml
```

Equivalent invocations:

```bash
python -m pid3net.train configs/Moving_chart_1ms.yaml
python train_ssp.py configs/Moving_chart_1ms.yaml   # legacy back-compat shim
```

By default this runs **20 epochs** with **5 refinement steps**, the
`multi_c` probe mode, MSE diffraction loss, and the `refractive`
reconstruction mode. Training output (checkpoints, history, and a
source-code archive) lands under `trained_models/<auto-suffixed-name>/`.

The auto-suffix encodes the run configuration, e.g.

```
trained_models/Moving_chart_1ms_3d3_mse_..._r5_multi_c_refractive_pie_seed0/
```

so multiple runs don't collide.

## Inference

Skip training and re-run inference from existing weights:

```bash
pid3net-train configs/Moving_chart_1ms.yaml --inference-only
```

Or use a specific checkpoint:

```bash
pid3net-train configs/Moving_chart_1ms.yaml \
    --inference-only \
    --pretrained trained_models/.../models/model_unsp.tf
```

## Common overrides

| Goal | Flag |
|---|---|
| 2D ablation | `--model 2d` |
| More refinement steps | `--n_refine 7` |
| RAAR instead of ePIE | `--update_method raar` |
| Poisson NLL loss | `--dist` |
| Reproducibility | `--seed 42` |
| Long run | `--epoch 50` |

See [Training](../guides/training.md) for the full CLI reference and
[Configuration](../guides/configuration.md) for the YAML reference.

## Inspect the output

After training, the run directory contains:

```
trained_models/<run-name>/
├── models/                    # Keras checkpoints (.tf)
├── hist_train.npy             # per-epoch loss history
├── object_reconstruction.npz  # final reconstructed object stack
├── pid3net.py                 # snapshot of the model source used
├── base_model.py              # snapshot of training loop
├── physics_layers.py          # snapshot of RefineLayer
└── losses/                    # snapshot of loss functions
```

A typical post-run analysis:

```python
import numpy as np
data = np.load("trained_models/<run-name>/object_reconstruction.npz")
amp, phase = data["amplitude"], data["phase"]   # shape [T, H, W]
```

## Next steps

- [Apply to a new dataset](new-dataset.md) — the recipe for your own data.
- [Architecture](../concepts/architecture.md) — what the model actually
  does between the input diffraction stack and the output object.
- [Refinement](../concepts/refinement.md) — design notes on the
  noise-aware modulus projection.
