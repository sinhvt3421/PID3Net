# Losses and regularisers

PID3Net's training objective combines:

1. A **diffraction-space loss** comparing predicted to measured
   diffraction intensity.
2. An optional **TV regulariser** on the reconstructed object volume.
3. An optional **prior-phase loss** comparing the reconstructed phase
   to an ODE-interpolated prior.

This page describes each. API reference lives in
[`pid3net.losses`](../api/losses.md).

## Diffraction-space losses (`pid3net.losses.diffraction`)

### `masked_SEloss` — default

Masked squared error on the **sqrt-intensity** (i.e. amplitude):

$$L \;=\; \frac{\sum_q m(q) \cdot \big(\sqrt{I_\text{pred}(q)} - \sqrt{I_\text{meas}(q)}\big)^2}{\sum_q m(q)}$$

with mask $m(q) = 1$ where $I_\text{meas}(q) \neq 0$, else 0.

Two reasons to operate on $\sqrt{I}$ rather than $I$:

- The sqrt is approximately variance-stabilising for Poisson noise
  ($\text{Var}(\sqrt{I}) \approx 1/4$ regardless of mean).
- It puts dim and bright pixels on a more equal footing — a pure-$I$
  MSE is dominated by the central beam.

### `negative_log_loss` and `negative_log_loss_t` — Poisson NLL

When `--dist` is passed, the model emits a TFP `Poisson` distribution
and the loss becomes the negative log-likelihood:

$$L \;=\; -\sum_q m(q) \cdot \log p\big(I_\text{meas}(q)\,\big|\,\lambda = I_\text{pred}(q)\big)$$

with the mask thresholding low-intensity pixels (where the Poisson
NLL is numerically unstable or noise-dominated). The two variants
differ only in how the mask is applied:

- `negative_log_loss_t(min_val=3.0)` — zero-out the *measurement* below
  threshold, then evaluate `-log_prob` over all pixels.
- `negative_log_loss(min_val=1.0)` — keep all measurements, but
  *weight* the per-pixel `-log_prob` by the mask before summing over
  spatial axes.

Both normalise by the magnitude of the maximum intensity so the loss
scale stays comparable across exposure times.

## Object-space TV regularisers (`pid3net.losses.regularizers`)

Total-variation smoothness on the reconstructed amplitude or phase
stack `[B, T, H, W]`. Three flavours:

### `total_var` — 2D anisotropic per-frame

$$\text{TV}_{2D}(O) \;=\; \frac{1}{T \cdot 2 \cdot H^2}\sum_{t,r}\big(|\partial_x O| + |\partial_y O|\big)$$

No temporal coupling.

### `total_var_3d` — 3D anisotropic

Same as above but adds an L1 temporal term:

$$+\; \frac{1}{2 (T-1) H^2}\sum_{t,r}|\partial_t O|$$

### `total_var_3d_iso` — 3D isotropic spatial + L1 temporal (default)

The default regulariser inside the `TV` layer
(`pid3net.layers.physics_layers.TV`):

$$\text{TV}_\text{iso}(O) \;=\; \frac{1}{H}\sum_{t,r}\sqrt{|\partial_x O|^2 + |\partial_y O|^2 + \varepsilon} \;+\; \frac{1}{H^2}\sum|\partial_t O|$$

Isotropic spatial norm penalises edges in all directions equally
(unlike anisotropic which has directional bias on axis-aligned edges);
the temporal L1 term keeps cross-frame smoothness sparse to avoid
penalising legitimate dynamics.

## Prior-phase loss

When `use_prior_phase: true`, `PriorPhaseLoss`
(`pid3net.layers.fusion.PriorPhaseLoss`) adds

$$\text{weight}(t)\cdot\langle(\,\phi_\text{recon} - \phi_\text{prior})^2\rangle$$

with the weight cosine-annealed across epochs from `lambda_prior`
(initial, default 10.0) to `lambda_prior_min` (final, default 1.0).
The annealing is driven by `PriorLossDecay` in
`pid3net.models.base_model`.

The prior itself is an ODE-interpolated high-exposure phase
reconstruction; the loss pulls the dynamic, low-exposure reconstruction
toward the static guidance early in training, then relaxes as the
network converges.

## Choosing a configuration

The default training configuration (MSE diffraction loss, isotropic 3D
TV regulariser, no prior) is a sensible starting point. Common
ablations:

1. Add Poisson NLL: `--dist`. Most useful when photon counts are low
   and the Poisson noise model is more accurate than Gaussian.
2. Enable the prior phase: `use_prior_phase: true` with a populated
   `phase_dir:`. Helps on dynamic samples where a high-exposure static
   reconstruction is available.
3. Tune the TV strength: edit `gama` in the `TV` layer's instantiation
   in `pid3net/models/pid3net.py`. The current default is moderate; too
   high blurs detail, too low admits speckle.
