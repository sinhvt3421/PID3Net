# Training

```
pid3net-train <dataset_config> [options]
```

## CLI flags

| Argument | Type | Default | Description |
|---|---|---|---|
| `dataset` | str | *required* | Path to dataset YAML config. |
| `--model` | str | `3d3` | Architecture: `3d3` (PID3Net) or `2d` (PIBaseD3Net ablation). |
| `--n_refine` | int | `5` | Number of refinement iterations. |
| `--probe_mode` | str | `multi_c` | One of `single`, `single_c`, `multi`, `multi_c`. |
| `--rec_mode` | str | `refractive` | `polar` or `refractive`. |
| `--update_method` | str | `pie` | `pie` (ePIE) or `raar`. |
| `--pretrained` | str | `""` | Path to a `.tf` checkpoint to warm-start from. |
| `--dist` | flag | off | Switch to Poisson NLL diffraction loss (default is masked MSE). |
| `--epoch` | int | `20` | Number of training epochs. |
| `--seed` | int | `0` | RNG seed (Python, NumPy, TF, `PYTHONHASHSEED`). |
| `--inference-only` | flag | off | Skip training; run inference using `--pretrained` weights. |

CLI flags override the corresponding keys in the YAML; everything
else is read from the YAML.

## Common training recipes

### Default

```bash
pid3net-train configs/Moving_chart_1ms.yaml
```

### Poisson NLL on photon-count data

```bash
pid3net-train configs/Moving_chart_1ms.yaml --dist
```

Use this when the diffraction stack is in actual photon counts (not
normalised amplitude) and the lowest-intensity pixels matter.

### RAAR instead of ePIE

```bash
pid3net-train configs/Moving_chart_1ms.yaml --update_method raar
```

### More refinement steps

```bash
pid3net-train configs/Moving_chart_1ms.yaml --n_refine 7
```

Doubles refinement loop cost; expect ~10 % wall-clock increase per
+1 step. Marginal quality gains beyond 5 in our experiments — see
[Refinement](../concepts/refinement.md).

### Reproducible run

```bash
pid3net-train configs/Moving_chart_1ms.yaml --seed 42
```

Seeds Python, NumPy, TF, and `PYTHONHASHSEED`. **GPU operations
involving `tf.signal.fft2d` are not bitwise deterministic across
runs**, so reproducibility is approximate at the loss level (~1e-5
relative noise).

### Resume from a previous run's weights

```bash
pid3net-train configs/Moving_chart_1ms.yaml \
    --pretrained trained_models/<prev_run>/models/model_unsp.tf
```

The model loads weights with `expect_partial()`, so partial matches
(e.g. when only the encoder/decoder shape matches) succeed without
erroring.

### Full ablation example

```bash
pid3net-train configs/Moving_chart_1ms.yaml \
    --model 3d3 \
    --probe_mode multi_c \
    --rec_mode refractive \
    --update_method pie \
    --n_refine 5 \
    --dist \
    --epoch 30 \
    --seed 7
```

## What gets written to disk

```
trained_models/<auto-suffixed-name>/
├── models/                       # Keras .tf checkpoints
├── hist_train.npy                # per-epoch loss history (dict)
├── object_reconstruction.npz     # final amplitude + phase stack
├── pid3net.py                    # source snapshot — model
├── base_model.py                 # source snapshot — training loop
├── physics_layers.py             # source snapshot — RefineLayer
└── losses/                       # source snapshot — losses subpackage
```

The source snapshots are written by `pid3net.train._archive_sources`
using package-relative paths, so they work whether you installed
editable, from a wheel, or from a source checkout.

## See also

- [Inference](inference.md) — run-only mode.
- [Configuration](configuration.md) — every YAML key.
- [Probe modes](probe-modes.md) — when to pick which.
