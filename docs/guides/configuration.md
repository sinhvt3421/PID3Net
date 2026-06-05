# Configuration reference

Every PID3Net run is driven by a YAML config with two top-level sections:
`model:` and `hyper:`. CLI flags override individual keys.

A documented template lives at `configs/_template.yaml`; copy it and
edit. This page is the canonical reference.

## `model:` section

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | str | `"3d3"` | Registry key. Also `--model`. One of `"3d3"` (PID3Net) or `"2d"` (PIBaseD3Net ablation). |
| `filters` | int | `8` | Encoder/decoder base channel count. |
| `n_layers` | int | `4` | Encoder/decoder depth. |
| `loss` | float | dataset-dependent | Loss-weight multiplier on the diffraction term. |

## `hyper:` section — core training

| Key | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `5e-4` | Adam learning rate. |
| `batch_size` | int | `1` | Frames per batch. |
| `epoch` | int | `20` | Number of epochs (CLI `--epoch` overrides). |
| `seed` | int | `0` | RNG seed. |
| `dist` | bool | `false` | Use Poisson NLL output (CLI `--dist`). |

## `hyper:` — physics

| Key | Type | Default | Description |
|---|---|---|---|
| `probe` | str | required | Path to probe `.npy`. Complex64. Shape `[H, W]` (single) or `[M, H, W]` (multi). |
| `probe_mode` | str | `"multi_c"` | One of `single`, `single_c`, `multi`, `multi_c`. See [Probe modes](probe-modes.md). |
| `probe_norm` | float / bool | `false` | Amplitude scaling on the probe (typically exposure-ratio). |
| `masking` | str / bool | `false` | Path to spatial mask `.npy`, or `false`. |
| `rec_mode` | str | `"refractive"` | `polar` (`A·exp(jφ)`) or `refractive` (`φ + j·A`). |

## `hyper:` — refinement

| Key | Type | Default | Description |
|---|---|---|---|
| `n_refine` | int | `5` | Number of refinement iterations. |
| `update_method` | str | `"pie"` | `pie` (ePIE) or `raar`. |
| `refine.poisson_projection.enabled` | bool | `false` | Opt-in: **replace** Gaussian-MLE projection with Poisson-MLE gradient step. See [Refinement](../concepts/refinement.md). |
| `refine.poisson_projection.eps` | float | `1e-3` | Tikhonov inside `I_meas / (\|Ψ\|² + ε)`. |

## `hyper:` — dataset

| Key | Type | Default | Description |
|---|---|---|---|
| `sample` | str | required | Loader key — must match a function in `pid3net.utils.general.dataset_functions`. |
| `train_data` | str | required | Path to the diffraction `.npz`. |
| `init_pty` | str / bool | `false` | Optional path to initial reconstruction `[2, H, W]`. Enables `TimeDecayFusion`. |

## `hyper:` — prior phase {#prior-phase}

| Key | Type | Default | Description |
|---|---|---|---|
| `use_prior_phase` | bool | `false` | Enable phase-prior fusion + annealing loss. |
| `use_prior_amp` | bool | `false` | Also include amplitude prior (usually leave off). |
| `phase_dir` | str / null | `null` | Directory containing per-step prior files. |
| `phase_file_pattern` | str | `"f{time:04d}.npy"` | Filename template. |
| `phase_key` | str | `"xhat"` | NPZ key, when loading from `.npz`. |
| `diff_dt_ms` | float | `1.0` | Diffraction frame interval (ms). |
| `phase_dt_ms` | float | `1.0` | Prior frame interval (ms). |
| `lambda_prior` | float | `10.0` | Initial prior-loss weight (cosine-annealed). |
| `lambda_prior_min` | float | `1.0` | Final prior-loss weight. |

## `hyper:` — output

| Key | Type | Default | Description |
|---|---|---|---|
| `save_path` | str | required | Output directory (auto-suffixed with run config). |
| `tvo` | bool | `false` | Enable TV regularisation on the reconstructed object. |

## Path resolution

Currently paths in the YAML are resolved against the **current working
directory**. Use absolute paths if you want `pid3net-train` to work
from any directory.

A `${DATA_ROOT}` environment-variable placeholder is planned but not
shipped in this release; see the [Changelog](../changelog.md) for
status.

## Example

```yaml
hyper:
  # physics
  probe: "/abs/path/probe.npy"
  probe_mode: "multi_c"
  probe_norm: 1.0
  masking: false

  # dataset
  sample: "chart"
  train_data: "/abs/path/frame_diff.npz"

  # training
  lr: 5.0e-4
  batch_size: 1
  loss: 2.0

  # refinement
  n_refine: 5
  update_method: "pie"
  rec_mode: "refractive"
  refine:
    poisson_projection:
      enabled: false

  # prior
  use_prior_phase: false

  # output
  save_path: "trained_models/my_run"
  tvo: false

model:
  model: "3d3"
  filters: 8
  n_layers: 4
  loss: 1.0
```
