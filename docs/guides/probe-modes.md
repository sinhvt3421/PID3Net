# Probe modes

The `probe_mode` flag controls how the illumination probe is treated by
the forward model and the refinement update.

## The four modes

| Mode | Probe shape | Forward model | Object update | When to use |
|---|---|---|---|---|
| `single` | `[1, H, W]` | `Ψ = F{P · O}` | Analytic gradient only. | Smallest / cheapest. Use when the probe is single-mode and you don't need a learned correction. |
| `single_c` | `[1, H, W]` | `Ψ = F{P · O}` | Analytic gradient **+ CNN nudge**. | Single-mode probe with learned regularisation. |
| `multi` | `[M, H, W]` | `I_pred = ∑ₘ \|F{Pₘ · O}\|²` | Analytic gradient summed over modes. | Multi-mode probe, physics-only update. |
| `multi_c` | `[M, H, W]` | `I_pred = ∑ₘ \|F{Pₘ · O}\|²` | Analytic + CNN nudge. **Default.** | Multi-mode probe with learned regularisation. |

## How to pick

- **Single-mode source** (well-focused, coherent X-ray): start with
  `single_c`.
- **Multi-mode source** (multi-mode FEL, partial coherence): start with
  `multi_c`.
- **Physics-only baseline / ablation**: use the corresponding
  non-`_c` variant. Useful for measuring how much the CNN actually
  contributes.

## How the multi-mode reduction works

For multi-mode probes the per-mode exit waves are
`ψₘ = Pₘ · O` (polar) or `ψₘ = Pₘ · exp(j·O)` (refractive).
The forward FFT runs per mode, and the measurable intensity is the
**incoherent sum**:

$$I_\text{pred}(q) \;=\; \sum_m |\mathcal{F}\{\psi_m\}(q)|^2$$

The modulus projection therefore acts on `sqrt(I_pred)`, and the
analytic gradient `conj(P) · dψ` is summed over modes before being
fed to the CNN. The CNN sees a per-pixel gradient, not a per-mode one
— inter-mode information is not present in the measurement, so the
symmetric reduction is the principled choice.

## Switching modes mid-project

The probe-mode choice influences:

- The shape of the loaded probe file (1 mode vs M modes).
- Whether the `CNNTBLayer` correction is instantiated.

So switching `multi_c` → `single_c` requires a re-train (CNN weights
don't transfer cleanly across mode dimensionality). Switching
`multi_c` → `multi` keeps the same probe file but discards the CNN
heads.

## See also

- [Physics](../concepts/physics.md) — the math of the forward model.
- [Refinement](../concepts/refinement.md) — what `dψ` and the CNN
  nudge actually do.
