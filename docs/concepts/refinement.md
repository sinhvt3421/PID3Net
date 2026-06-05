# Refinement design log

This page consolidates the reasoning behind PID3Net's refinement block
and the noise-aware modulus projection options. It is written for
researchers who want to understand *why* the code is the way it is, not
just *what* it does.

## The problem

`RefineLayer.apply_intensity_constraint` originally hard-replaced the
predicted modulus with the measured one:

```python
return tf.where(
    org_intensity >= 0,
    tf.cast(org_intensity / intensity, "complex64") * dif,
    dif,
)
```

This is the *Gaussian-noise maximum-likelihood projection* — the
correct one for additive Gaussian noise. **For low-count Poisson
pixels** (typical of high-q diffraction in dynamic CXDI) **it is
biased**: a pixel with measured count 0 is forced to predict
$|\Psi| = 0$, even if the true mean is 1–2 photons.

The bias propagates through the rest of the pipeline:

$$d\psi = \psi' - \psi \;\Longrightarrow\; P^* \cdot d\psi \;\Longrightarrow\; \text{CNN} \;\Longrightarrow\; \text{object update.}$$

Every downstream component is polishing a noise-naive substrate.

## Rejected alternatives

These were considered and dropped early, in roughly increasing order of
appeal:

| Candidate | Reason for rejection |
|---|---|
| Per-step CNN weights | The physics-imposed $P^*$ already weighs pixels by probe confidence — extra sigmoids on top are redundant. |
| Richer CNN inputs (residual magnitude, probe norm, step embedding) | Same: adds parameters without adding information content. |
| Per-pixel learned $\alpha$ map | Re-implements what $P^*$ already provides. |
| Highway-style update gate | Same — sigmoid on top of a noise-naive gradient. |
| ConvGRU / RIM-style memory | Integrates noisy gradients across steps but still consumes a noisy substrate. Treats symptom, not source. |
| Plug-and-Play / DEQ projection | Discards the analytic physics signal. Data-hungry, hallucination-prone, weak ptychography precedent. |
| Edge-preserving / Charbonnier / TV prior loss | Operates downstream of the noise; cannot recover information already lost. |
| Per-pixel low-intensity loss weighting | Same — re-weights what is already corrupted. |
| Time-averaged surrogate `I_high = mean_t(I_meas²)` as projection target | Only suppresses the temporal DC component; loses any moving structure (next paragraph). |
| Learned temporal Wiener filter on `I_meas` | Breaks Hermitian symmetry of `FFT_T(I_meas)`; empirically regressed reconstruction quality on every dataset tested. |

### The earlier surrogate attempt

A learned per-pixel fusion between per-frame `I_meas` and a static
time-averaged surrogate `I_high = mean_t(I_meas²)`:

```python
mix_intensity = sqrt(
    (1 - σ(β)) * I_meas² + σ(β) * I_high + ε
)
```

In temporal-frequency terms, `I_high` is the **DC component along T**
only. The mix is a two-tap filter responding at $\omega = 0$ and
"all-pass", with nothing between. Real dynamic signal lives at
low-but-nonzero $\omega$. The mix could only denoise the *static
background*; it could not denoise the *moving structure*. Empirical
result matched: small improvement, then plateau.

## Final design

One opt-in change inside `RefineLayer.apply_intensity_constraint`.
`RefineLayer.call`'s signature is unchanged.

### Poisson-MLE projection (replaces Gaussian when enabled)

Hard modulus replacement is the Gaussian-noise MLE. The Poisson
log-likelihood has no closed-form modulus update; we take one gradient
step per refinement iteration:

$$\nabla_{\!\Psi} L_\text{Poisson} \;=\; \Psi \cdot \left(1 - \frac{I_\text{meas}}{|\Psi|^2 + \varepsilon}\right)$$

$$\Psi \;\leftarrow\; \Psi \;-\; \eta\cdot\nabla_{\!\Psi} L_\text{Poisson}$$

with $\eta$ a per-pixel SNR-dependent step (the implementation uses
$\eta = I_\text{meas} / (I_\text{meas} + \text{noise\_floor})$ so the
update is small on low-count pixels). The outer refinement loop applies
this update `n_step` times so the effect accumulates without inner
iterations.

Mutually exclusive with the Gaussian projection — a clean switch, not a
blend. An earlier blended design with a learnable $\rho$ was rejected
because it gave the optimiser too much freedom to fall back to the
biased Gaussian projection rather than committing to the Poisson
formulation.

**Behaviour at the failure mode.** When $I_\text{meas} = 0$ but the
true mean is nonzero, the Gaussian rule sets $|\Psi| = 0$
(irrecoverable). The Poisson correction is
$1 - 0/(|\Psi|^2 + \varepsilon) = 1$, so it merely *shrinks* $\Psi$
by $(1 - \eta)$ rather than zeroing it — leaving room for later
iterations to recover. At the fixed point $|\Psi|^2 = I_\text{meas}$
the correction term vanishes and the update is a no-op, matching the
Gaussian projection's fixed point.

## Empirical status (as of v2.0.0)

- Shipped but **defaults off** — existing configs are unaffected.
- On the datasets tested so far, the Poisson projection has shown
  parity with or slightly below the Gaussian projection. It is
  retained as an opt-in for low-count / photon-starved datasets
  where the Gaussian projection's zero-pixel bias is most visible.

## References

- Fienup, J. R. **Phase retrieval algorithms: a comparison.** *Appl.
  Opt.* 21, 2758 (1982). Origin of modulus replacement; assumes
  Gaussian noise.
- Thibault, P. & Guizar-Sicairos, M. **Maximum-likelihood refinement
  for coherent diffractive imaging.** *Phys. Rev. Lett.* 109, 068101
  (2012). Poisson MLE formulation for ptychography — basis for the
  optional Poisson projection.
- Godard, P. *et al.* *Opt. Express* 20 (2012). Poisson likelihood for
  ptychography, alternative derivation.
