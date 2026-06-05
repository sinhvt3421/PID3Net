# Apply to a new dataset

PID3Net needs three things to train on your data: a **diffraction stack**,
a **probe function**, and a **YAML config** pointing at them.

## 1. Prepare the data files

| File | Format | Shape | Notes |
|---|---|---|---|
| Diffraction stack | `.npz` | `[T, H, W]` float32 | Per-frame measured intensity *or* amplitude, depending on the loader. Conventionally amplitude `sqrt(I)`. |
| Probe function | `.npy` complex64 | `[H, W]` (single) or `[M, H, W]` (multi-mode) | Pre-calibrated probe — usually from a one-shot static reconstruction. |
| Spatial mask (optional) | `.npy` float | `[H, W]` | Binary or soft; `1.0` = trusted, `0.0` = ignore. |
| Init reconstruction (optional) | `.npy` | `[2, H, W]` float32 | `[amplitude_init, phase_init]`, used by `TimeDecayFusion`. |
| Phase prior (optional) | `.npy` per timestep | `[H, W]` float | ODE-interpolated phase prior frames; see [Configuration](../guides/configuration.md#prior-phase). |

Pick whichever existing dataset loader most closely matches your data
shape (see `pid3net.utils.general.dataset_functions`), or register a new
one. The loaders live in `pid3net/utils/general.py`.

## 2. Copy the template config

```bash
cp configs/_template.yaml configs/my_experiment.yaml
$EDITOR configs/my_experiment.yaml
```

The template has every key inline-commented. The minimum to set:

```yaml
hyper:
  probe: "/abs/path/to/probe.npy"
  train_data: "/abs/path/to/diffraction.npz"
  sample: "chart"                            # which loader to use
  save_path: "trained_models/my_experiment"
model:
  model: "3d3"                               # PID3Net default
```

Everything else has sensible defaults — start with those, ablate from
there.

## 3. Train

```bash
pid3net-train configs/my_experiment.yaml
```

Override individual keys on the command line:

```bash
pid3net-train configs/my_experiment.yaml \
    --n_refine 7 \
    --update_method raar \
    --dist
```

## 4. Iterate

A typical sweep flow:

1. Confirm reconstruction quality with the defaults.
2. Try `--update_method raar` if PIE underconverges on low-SNR pixels.
3. Try `--n_refine 7` or `10` (slower; trade-offs).
4. Try `--dist` for Poisson NLL on photon-count diffraction.
5. If complex regions blur, see the noise-aware projection options
   under `refine.*` in the YAML — they're opt-in and documented in
   [Configuration](../guides/configuration.md).

## Adding a brand-new sample type

If none of the existing loaders fits your data shape:

1. Write a new loader in `pid3net/utils/general.py` returning the same
   tuple shape as the existing ones.
2. Register it in the `dataset_functions` dict at the bottom of that file.
3. Reference its key from your config's `sample:` field.

No other code change is required.
