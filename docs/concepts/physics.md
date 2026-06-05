# Physics

This page explains the physical content of the refinement block:
what the exit wave and modulus projection mean, why two reconstruction
modes exist, and how PIE differs from RAAR.

## Forward model

The exit wave at scan position $j$ is the probe times the object
transmittance:

$$\psi_j(r) \;=\; P(r - r_j)\cdot O(r)$$

The far-field diffraction amplitude is its scaled FFT:

$$\Psi_j(q) \;=\; \mathcal{F}\{\psi_j\}(q)$$

The detector measures intensity:

$$I_j(q) \;=\; |\Psi_j(q)|^2$$

For *multi-mode* probes (incoherent), the measured intensity is the
sum over modes:

$$I_j(q) \;=\; \sum_m |P_m \cdot O|^2$$

## Polar vs refractive

Two parameterisations of the complex object are supported:

=== "Polar (default in classical ptychography)"

    $$O(r) \;=\; A(r)\cdot e^{j\phi(r)}$$

    Amplitude $A \in [0, 1]$ and phase $\phi \in [-\pi, \pi]$ are
    independent. Used when the sample is opaque (large amplitude
    contrast).

=== "Refractive (default in PID3Net)"

    $$O(r) \;=\; \phi(r) + j\cdot A(r)$$

    Real part is the refractive index decrement (phase shift),
    imaginary part is the absorption. Linear in $(\phi, A)$, more
    natural for weakly-absorbing samples. The
    `AmpConstraint` activation clips $A$ to $[-0.5, 5.0]$.

The `rec_mode:` config key (or `--rec_mode` CLI flag) picks one.

## ePIE (extended Ptychographic Iterative Engine)

Per iteration:

1. **Exit wave** $\psi = P \cdot O$ (or $\psi = P \cdot e^{j O}$ refractive).
2. **Forward FFT** $\Psi = \mathcal{F}\{\psi\}$.
3. **Modulus projection** $\Psi' = \sqrt{I_\text{meas}} \cdot \Psi/|\Psi|$
   (Gaussian-MLE; Poisson-MLE).
4. **Inverse FFT** $\psi' = \mathcal{F}^{-1}\{\Psi'\}$.
5. **Object update** along the gradient of the data-fit loss:

    $$\nabla_O L \;=\; -P^* \cdot (\psi' - \psi)$$

    so

    $$O \;\leftarrow\; O \;+\; \alpha\cdot \frac{P^*}{\max|P|^2}\cdot d\psi.$$

    In PID3Net the gradient is decomposed into amplitude and phase
    components and passed through a small CNN
    (`pid3net.layers.physics_layers.CNNTBLayer`) before being added.

The conjugate $P^*$ is doing two things at once:

- **Phase alignment** – undoes the probe's local phase so the residual
  lands in the object's coordinate system.
- **Probe-amplitude gating** – multiplies by $|P|$, so pixels where the
  probe is dim receive near-zero updates. This is physics-imposed
  per-pixel confidence weighting.

## RAAR (Relaxed Averaged Alternating Reflections)

Instead of a single projection–update pair, RAAR uses *double reflection*
through both the Fourier (data) and overlap (object) projection sets:

$$\psi^{(\text{RF})} \;=\; 2\,\psi' \;-\; \psi$$

$$\psi^{(\text{RS RF})} \;=\; 2\,\mathcal{P}_S(\psi^{(\text{RF})}) \;-\; \psi^{(\text{RF})}$$

$$\psi^{(\text{RAAR})} \;=\; \tfrac{\beta}{2}\big(\psi^{(\text{RS RF})} + \psi\big) \;+\; (1-\beta)\,\psi'$$

with $\beta \in [0.5, 1]$ a learned relaxation parameter
($\beta = 1$ recovers Difference-Map; small $\beta$ approaches
alternating projections). RAAR explores the solution space more broadly
than PIE and is *in principle* more robust to noise — though in
practice that benefit depends strongly on the dataset, and PID3Net
often performs better with PIE.

Selected via `--update_method pie | raar`.

## Probe modes

| Mode | Probe shape | Constraint application | Object update |
|---|---|---|---|
| `single_c` | `[1, H, W]` | per-pixel modulus replacement | analytic gradient + CNN nudge |
| `multi_c` | `[M, H, W]` | summed intensity per-pixel | analytic gradient + CNN nudge (default) |

Both modes route the analytic gradient through a learned `CNNTBLayer`
correction. The physics-only (`single` / `multi`, without the `_c`
suffix) variants from v1.0.0 are not part of the v2 public surface.

## Further reading

- Fienup 1982 — origin of the modulus replacement (Gaussian-MLE) for
  phase retrieval.
- Thibault & Guizar-Sicairos PRL 2012 — Poisson-MLE refinement for
  ptychography; basis for the optional Poisson projection.
- Maiden & Rodenburg 2009 — extended PIE (ePIE).
- Luke 2005 — Relaxed Averaged Alternating Reflections.
