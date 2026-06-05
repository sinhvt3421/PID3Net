# Architecture

PID3Net's default `PID3Net` model is a **3D temporal encoder–decoder with a
physics-informed iterative refinement block**. Three baseline variants
trade dimensions or refinement for simpler comparisons.

```
      　┌──────────┐    ┌──────────┐
diff ─▶　│ TBEncoder│ ─▶│ TBDecoder│─┐
        └──────────┘    └──────────┘ │
                                             ▼
                                    ┌─────────────────┐
   optional init/prior ────────▶ 　　│PriorFusion 　　│
                                    └─────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ CombineComplex  │
                                    └─────────────────┘
                                             │
                                             ▼
               measured I ──▶ ┌──────────────────────┐
                              │     RefineLayer      │
               probe ────────▶│  (n_refine × ePIE /  │
                              │   RAAR + CNN nudges) │
                              └──────────────────────┘
                                             │
                                             ▼
                                    output head
                                 (diff_intensity,
                                 refined amplitude,
                                 refined phase)
```

## Components

### Encoder – `TBEncoder`

`pid3net.layers.encoders.TBEncoder` (used by `PID3Net`) is a temporal-block
encoder operating on input shape `[B, T, H, W, 1]`. It stacks
`Conv_Down_Temporal_Block`s with growing channel counts
(`filters * 2**i`) and an internal latent block at the bottom.

The 2D baseline (`PIBaseD3Net`) uses `CNNEncoder` instead, operating
on `[B, H, W, 1]` without the temporal dimension.

### Decoders – dual `TBDecoder`

Two parallel `TBDecoder`s reconstruct **amplitude** and **phase**
separately, mirroring the encoder's depth. Output activations:

- Polar mode: `sigmoid` amplitude, `Mpi` phase (bounded to ±π).
- Refractive mode: `AmpConstraint` amplitude (bounded to [−0.5, 5.0]),
  linear phase.

### Optional fusion blocks

- **`TimeDecayFusion`** – when `init_pty:` is set, blends the decoded
  estimate with an external initial reconstruction, with a time-decaying
  weight (higher trust on the prior at t=0).
- **`PriorPhaseFusion`** – per-pixel learned blend between the decoder
  output and an ODE-interpolated phase prior, enabled via
  `use_prior_phase: true`.

### `CombineComplex`

Assembles the complex object from amplitude and phase tensors:

- Polar: `amp · exp(j·φ)`
- Refractive: `φ + j·amp`

### `RefineLayer`

The physics-informed iterative refinement block, the heart of the model.
For each of `n_refine` steps:

1. Form the **exit wave** `ψ = P · O` (polar) or `ψ = P · exp(j·O)`
   (refractive).
2. Far-field FFT to get the predicted diffraction `Ψ`.
3. **Intensity constraint** – the *modulus projection*. Default:
   hard Gaussian-MLE replacement `Ψ ← sqrt(I_meas) · Ψ/|Ψ|`.
   Optionally Poisson-MLE (opt-in via `refine.poisson_projection`).
   See [Refinement](refinement.md).
4. Inverse FFT to get the constrained exit wave `ψ'`.
5. Object update from `dψ = ψ' − ψ`:
    - **ePIE** (default): `O ← O + α · CNN(decompose(P*·dψ))`.
    - **RAAR**: double reflection through Fourier + overlap projections,
      then the same CNN-modulated update.

After `n_refine` iterations the layer returns both the final predicted
diffraction amplitude and the refined complex object.

### Output head

Three outputs:

1. `diff_intensity` – `|Ψ_refined|²`, against which the diffraction loss
   is computed.
2. `amplitude_refined` – the refined amplitude tensor.
3. `phase_refined` – the refined phase tensor.

## Model variants

| Key | Class | Dimensions | Refinement | Probe modes | Notes |
|---|---|---|---|---|---|
| `3d3` | `PID3Net` | 3D temporal | ✅ | all four | Default, recommended. |
| `2d` | `PIBaseD3Net` | 2D spatial | ✅ | all four | 2D ablation — drops the temporal axis. Useful for measuring the contribution of T. |

Both variants are registered in `pid3net.models.MODEL_REGISTRY`; switch
by passing `--model 2d` on the CLI.

## Adding a new variant

See [Adding a model](../guides/adding-a-model.md) for the registry
pattern. The short version: write a new `PtyBase` subclass, set
`is_temporal = True|False` as a class attribute, then add one entry
to `MODEL_REGISTRY`. No edits to `base_model.py` or `train.py`.
