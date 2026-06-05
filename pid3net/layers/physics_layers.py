"""Physics-informed layers for ptychographic reconstruction.

Contains the iterative FFT-based refinement layer (RefineLayer), complex object
construction (CombineComplex), total variation regularization (TV), and CNN-based
update layers (CNNTBLayer, FusionLayer) used in the refinement loop.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Activation
from tensorflow.signal import fft2d, fftshift, ifftshift, ifft2d

from pid3net.losses import total_var_3d, total_var_3d_iso
from pid3net.layers.activations import Mpi, AmpConstraint
from pid3net.layers.conv_blocks import Conv_Down_Temporal_Block


def combine_complex(amp: tf.Tensor, phi: tf.Tensor, mode: str = "polar") -> tf.Tensor:
    """Combine amplitude and phase into a complex tensor.

    Args:
        amp: Amplitude tensor.
        phi: Phase tensor.
        mode: 'polar' for amp*exp(j*phi), 'refractive' for phi + j*amp.
    """
    return (
        tf.cast(amp, tf.complex64) * tf.exp(1j * tf.cast(phi, tf.complex64))
        if mode == "polar"
        else tf.cast(phi, tf.complex64) + 1j * tf.cast(amp, tf.complex64)
    )


class CombineComplex(tf.keras.layers.Layer):
    """Keras layer wrapping combine_complex for use in functional model graphs."""

    def call(self, amp: tf.Tensor, phi: tf.Tensor, mode: str = "polar") -> tf.Tensor:
        return combine_complex(amp, phi, mode=mode)


class TV(tf.keras.layers.Layer):
    """Total variation regularization layer.

    Adds a weighted 3D isotropic TV loss term without modifying the tensor.
    The weight (gamma) is optionally trainable and clipped to [0.01, 3.0].

    Args:
        gama: Initial TV regularization weight.
        name: Layer name.
        train: Whether gamma is trainable.
    """

    def __init__(self, gama: float, name: str = "TV", train: bool = False, **kwargs: object) -> None:
        super(TV, self).__init__(name=name, **kwargs)
        self.gama = tf.Variable(
            gama,
            name="gama_{}".format(name),
            trainable=train,
            constraint=lambda x: tf.clip_by_value(x, 0.01, 3.0),
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.add_loss(self.gama * total_var_3d_iso(inputs))
        return inputs


class CNNTBLayer(tf.keras.layers.Layer):
    """Small temporal-block CNN for learning object update corrections in refinement.

    Used in CNN-based probe modes (single_c, multi_c) to learn amplitude and phase
    update maps from the analytic gradient signal.

    Args:
        nfilters: Number of convolutional filters.
        w: Spatial kernel size.
        dept: Number of Conv_Down_Temporal_Block layers.
        act: Activation function for intermediate layers.
        out: Output activation: 'sigmoid', 'mpi', 'const', or '' (linear).
    """

    def __init__(
        self,
        nfilters: int = 32,
        w: int = 3,
        dept: int = 1,
        act: str = "swish",
        out: str = "sigmoid",
        name: str = "",
        **kwargs: object,
    ) -> None:
        super(CNNTBLayer, self).__init__(name=name, **kwargs)
        self.cv = [Conv_Down_Temporal_Block(nfilters, w, padding="same", act=act, pool=False) for i in range(dept)]
        self.cv_out = Conv3D(1, (1, w, w), padding="same", activation=None)

        if out == "sigmoid":
            self.act_out = Activation("sigmoid")
        elif out == "mpi":
            self.act_out = Mpi()
        elif out == "const":
            self.act_out = AmpConstraint()
        else:
            self.act_out = Activation("linear")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if len(tf.shape(inputs)) == 4:
            x = tf.expand_dims(inputs, -1)
        else:
            x = inputs
        for i in range(len(self.cv)):
            x = self.cv[i](x)
        return tf.squeeze(self.act_out(self.cv_out(x)), -1)


def ptychography_forward(
    objects: tf.Tensor,
    probe: tf.Tensor,
    probe_size: int,
    probe_mode: str = "single",
    refractive: bool = True,
) -> tf.Tensor:
    """Compute diffraction amplitude from a complex object (forward pass only, no refinement).

    Mirrors ``RefineLayer.compute_output_intensity`` so the forward model can be
    evaluated independently — e.g. to recompute ``diff_amp_r`` outside the
    refinement loop.

    Args:
        objects: Complex object tensor ``[B, T, H, W]`` (complex64).
        probe: Probe tensor — ``[1, H_p, W_p]`` for single-mode probes,
            ``[M, 1, H_p, W_p]`` for multi-mode probes.
        probe_size: Integer spatial probe size used as the FFT normalisation constant.
        probe_mode: Probe mode string, e.g. ``"single"``, ``"single_c"``,
            ``"multi"``, ``"multi_c"``.
        refractive: If True, the object is treated as ``phi + j*amp`` (refractive
            index mode).  If False, the object is the complex transmittance (polar).

    Returns:
        Diffraction amplitude ``[B, T, H, W]`` (float32).
    """
    if "single" in probe_mode:
        exit_wave = probe * (tf.exp(1j * objects) if refractive else objects)
        dif = fftshift(fft2d(exit_wave), axes=(-2, -1)) / probe_size
        return tf.abs(dif)
    else:
        objects_exp = tf.exp(1j * tf.expand_dims(objects, 1)) if refractive else tf.expand_dims(objects, 1)
        exit_wave = probe * objects_exp
        dif = fftshift(fft2d(exit_wave), axes=(-2, -1)) / probe_size
        return tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, axis=1))


class RefineLayer(tf.keras.layers.Layer):
    """Iterative physics-informed refinement layer for ptychographic reconstruction.

    Implements differentiable iterative phase retrieval with selectable update
    methods: ePIE (extended Ptychographic Iterative Engine) or RAAR (Relaxed
    Averaged Alternating Reflections).

    Each iteration computes exit wave -> forward FFT -> intensity constraint ->
    inverse FFT -> object update. Supports four probe modes (single, single_c,
    multi, multi_c) with optional CNN-learned updates and refractive index mode.

    RAAR uses double reflection through both Fourier and overlap constraint sets,
    which explores the solution space more broadly and is more robust to noisy
    or low-SNR measurements than PIE.

    Args:
        mask: Spatial mask tensor or None.
        n_step: Number of refinement iterations.
        probe_mode: Probe handling mode ('single', 'single_c', 'multi', 'multi_c').
        refractive: Use refractive index mode (object = phi + j*amp).
        update_method: Update rule — 'pie' (default) or 'raar'.
        refine_cfg: Optional dict of noise-aware projection options.  Recognised
            keys (all default off):

            - ``poisson_projection.enabled`` (bool, default False): *replace*
                the hard Gaussian-MLE modulus projection with a Poisson-MLE
                gradient step.  Mutually exclusive with the default Gaussian
                projection (no blending — pick one).
            - ``poisson_projection.eps`` (float, default 1e-3): Tikhonov
                regulariser inside ``I_meas/(|Psi|^2 + eps)``.

            When all flags are off the projection is bit-identical to the
            classical hard modulus replacement.
    """

    def __init__(
        self,
        mask: object,
        n_step: int = 5,
        probe_mode: str = "multi_c",
        refractive: bool = False,
        update_method: str = "pie",
        refine_cfg: object = None,
        **kwargs: object,
    ) -> None:
        super(RefineLayer, self).__init__(**kwargs)
        self.mask = mask

        if n_step > 0:
            self.alpha = tf.Variable([0.2] * n_step, trainable=True, dtype="float32", name="alpha")

        self.n_step = n_step
        self.probe_mode = probe_mode
        self.refractive = refractive
        self.update_method = update_method

        if "c" in self.probe_mode:
            self.cnn_tb_a = CNNTBLayer(out="const") if self.refractive else CNNTBLayer(out="sigmoid")
            self.cnn_tb_p = CNNTBLayer(out="") if self.refractive else CNNTBLayer(out="mpi")

        # ---- Noise-aware modulus projection options -------------------------
        refine_cfg = dict(refine_cfg or {})
        poiss_cfg = dict(refine_cfg.get("poisson_projection", {}) or {})

        self.use_poisson_projection = bool(poiss_cfg.get("enabled", False))

    def call(
        self,
        objects: tf.Tensor,
        org_intensity: tf.Tensor,
        probe: tf.Tensor,
        fftconst: int,
    ) -> "tuple[tf.Tensor, tf.Tensor]":
        """Run n_step refinement iterations and return final diffraction amplitude.

        Args:
            objects: Complex object tensor from encoder-decoder.
            org_intensity: Measured diffraction intensity (sqrt) for constraint.
            probe: Probe function tensor (complex64).
            fftconst: FFT normalization constant (probe spatial size).

        Returns:
            Tuple of (diffraction_amplitude, refined_objects).
        """
        prob_tf_abs = self.compute_probe_normalization(probe)

        for i in range(self.n_step):
            pre_exit = self.compute_exit_wave(objects, probe)

            dif = self.forward_fft(pre_exit, fftconst)

            dif = self.apply_intensity_constraint(dif, org_intensity, pre_exit)

            exitw = self.inverse_fft(dif, fftconst)

            objects = self.update_object(objects, exitw, pre_exit, probe, prob_tf_abs, iter=i)

        intensity = self.compute_output_intensity(objects, probe, fftconst)

        return intensity, objects

    def compute_probe_normalization(self, probe: tf.Tensor) -> tf.Tensor:
        """Compute max probe intensity for normalizing the object update step."""
        if "single" in self.probe_mode:
            return tf.cast(tf.reduce_max(tf.abs(probe) ** 2.0), "complex64")
        else:
            return tf.cast(tf.reduce_sum(tf.reduce_max(tf.abs(probe) ** 2, axis=(-2, -1)), 0), "complex64")

    def compute_exit_wave(self, objects: tf.Tensor, probe: tf.Tensor) -> tf.Tensor:
        """Multiply probe by object transmittance to get the exit wave."""
        if "single" in self.probe_mode:
            return probe * tf.exp(1j * objects) if self.refractive else probe * objects
        else:
            return (
                probe * tf.exp(1j * tf.expand_dims(objects, 1))
                if self.refractive
                else probe * tf.expand_dims(objects, 1)
            )

    def forward_fft(self, pre_exit: tf.Tensor, fftconst: int) -> tf.Tensor:
        """Propagate exit wave to far-field via shifted FFT."""
        return fftshift(fft2d(pre_exit), axes=(-2, -1)) / fftconst

    def apply_intensity_constraint(
        self, dif: tf.Tensor, org_intensity: tf.Tensor, pre_exit: tf.Tensor
    ) -> tf.Tensor:
        """Project predicted far-field amplitude onto the measurement.

        Exactly one of two projection rules is applied to ``dif``:

        * **Gaussian-MLE (default)**: hard modulus replacement
          ``Psi <- sqrt(I_target) * Psi/|Psi|``.  This is the closed-form MLE
        when the measurement noise is Gaussian.  Exactly equivalent to
        classical ePIE / DM.
        * **Poisson-MLE** (``refine.poisson_projection.enabled``): one
        gradient step of the Poisson log-likelihood,
          ``Psi <- Psi - eta * Psi * (1 - I_target/(|Psi|^2 + eps))``.
        The full refinement loop applies it
        ``n_step`` times so the effect accumulates.

        Args:
            dif: Predicted complex far-field ``Psi``.
            org_intensity: Measured diffraction amplitude (sqrt of intensity)
                ``[B, T, H, W]``.
            pre_exit: Pre-constraint exit wave (unused; kept for signature
                compatibility).

        Returns:
            Constrained far-field tensor with the same shape as ``dif``.
        """
        # ---- 1. Apply exactly one projection rule ---------------------------
        if self.use_poisson_projection:
            return self._poisson_projection(dif, org_intensity)
        return self._gaussian_projection(dif, org_intensity)

    def _gaussian_projection(
        self,
        dif,
        org_intensity,
    ):
        """Hard modulus replacement — the Gaussian-noise MLE.

        Rescales ``Psi(q) -> sqrt(I_target(q)) * Psi(q)/|Psi(q)|`` so that
        ``|Psi'| = sqrt(I_target)`` exactly, leaving the phase of ``Psi``
        untouched.  Valid pixels (``org_intensity >= 0``) only; invalid
        pixels pass through unchanged.

        The Gaussian-MLE projection assumes additive Gaussian noise on the
        amplitude.  For Poisson photon-count data with low counts it is
        biased: a pixel with measured count 0 is forced to predict
        ``|Psi| = 0`` even though the true mean might be 1-2 photons.
        """
        if "single" in self.probe_mode:
            intensity = tf.abs(dif)
            corr = tf.cast(org_intensity / (intensity + 1e-12), "complex64")
            return tf.where(org_intensity >= 0, corr * dif, dif)
        intensity = tf.sqrt(tf.reduce_sum(tf.abs(dif) ** 2, 1) + 1e-12)
        corr = tf.expand_dims(tf.cast(org_intensity / intensity, "complex64"), 1)
        return corr * dif

    def _poisson_projection(
        self, dif: tf.Tensor, org_intensity: tf.Tensor, noise_floor: float = 1.0
    ) -> tf.Tensor:
        """One gradient step of the Poisson log-likelihood.

        For Poisson observations with mean ``lambda(q) = |Psi(q)|^2``, the
        negative log-likelihood (up to constants) is

            L(Psi) = sum_q |Psi(q)|^2 - I_meas(q) * log(|Psi(q)|^2)

        and its Wirtinger gradient w.r.t. Psi is

            dL/dPsi = Psi * (1 - I_meas / |Psi|^2)

        A single gradient-descent step is

            Psi <- Psi - eta * Psi * (1 - I_meas / (|Psi|^2 + eps)).

        Why this matters for low-count pixels: when ``I_meas = 0`` but the
        true mean is nonzero, the Gaussian rule sets ``|Psi| = 0``
        (irrecoverably wrong).  The Poisson rule's correction term
        ``(1 - 0/(|Psi|^2 + eps))`` equals 1, so it merely *shrinks* ``Psi``
        by a factor ``(1 - eta)`` rather than zeroing it — leaving room for
        later iterations and probe-aware regularisation to recover the
        signal.  Conversely when ``I_meas = |Psi|^2`` (model matches data)
        the correction term vanishes and Psi is unchanged — the same fixed
        point as the Gaussian projection.

        """
        # I_meas: per-pixel measured target intensity.
        I_meas = tf.square(org_intensity)
        eta = I_meas / (I_meas + noise_floor + 1e-12)

        if "single" in self.probe_mode:
            # Single-probe case: |Psi|^2 is just |dif|^2.
            I_model = tf.square(tf.abs(dif))
            # Correction term: 1 - I_meas / (|Psi|^2 + eps).
            corr = 1.0 - I_meas / (I_model + 1e-12)
            corr_c = tf.cast(eta * corr, "complex64")
            # Gradient step.  Invalid pixels (org_intensity < 0) bypass the
            # update entirely so they remain at the prediction.
            updated = dif - dif * corr_c
            return tf.where(org_intensity >= 0, updated, dif)
        # Multi-mode case: |Psi|^2 = sum_m |dif_m|^2 over the probe-mode axis.
        I_model = tf.reduce_sum(tf.abs(dif) ** 2, 1)
        corr = 1.0 - I_meas / (I_model + 1e-12)
        # Broadcast the per-pixel correction over the mode axis so each
        # probe mode receives the same scalar nudge — preserves the relative
        # mode amplitudes (no inter-mode information is in I_meas).
        corr_c = tf.expand_dims(tf.cast(eta * corr, "complex64"), 1)
        return dif - dif * corr_c

    def inverse_fft(self, dif: tf.Tensor, fftconst: int) -> tf.Tensor:
        """Propagate constrained far-field back to real space via inverse FFT."""
        return ifft2d(ifftshift(dif, axes=(-2, -1))) * fftconst

    def compute_gradient(
        self,
        dexit: tf.Tensor,
        pre_exit: tf.Tensor,
        probe: tf.Tensor,
        prob_tf_abs: tf.Tensor,
        reduce_modes: bool = False,
    ) -> tf.Tensor:
        """Compute the object update gradient, aware of polar vs refractive mode.

        In refractive mode the exit wave is probe*exp(j*object), so d(exit)/d(object) = j*exit.
        In polar mode the exit wave is probe*object, so d(exit)/d(object) = probe.

        Args:
            dexit: Exit wave difference (constrained - predicted).
            pre_exit: Pre-constraint exit wave.
            probe: Probe function tensor.
            prob_tf_abs: Probe normalization factor.
            reduce_modes: If True, sum over probe mode dimension (axis=1) for multi-mode probes.
        """
        if self.refractive:
            numerator = tf.math.conj(1j * pre_exit) * dexit
            if reduce_modes:
                # Multi-mode: sum numerator over modes, normalize by summed peak intensities
                denominator = tf.cast(
                    tf.reduce_sum(tf.reduce_max(tf.abs(pre_exit) ** 2, axis=(-1, -2), keepdims=True), 1),
                    "complex64",
                )
                return tf.reduce_sum(numerator, 1) / denominator
            else:
                # Single probe: normalize by peak exit wave intensity
                denominator = tf.cast(
                    tf.reduce_max(tf.abs(pre_exit) ** 2, axis=(-1, -2), keepdims=True),
                    "complex64",
                )
                return numerator / denominator
        else:
            if reduce_modes:
                return tf.reduce_sum(tf.math.conj(probe) * dexit, 1) / prob_tf_abs
            return tf.math.conj(probe) * dexit / prob_tf_abs

    def decompose_gradient(self, invert_update: tf.Tensor) -> "tuple[tf.Tensor, tf.Tensor]":
        """Split complex gradient into amplitude and phase components.

        Refractive mode uses real/imag decomposition; polar uses abs/angle.
        """
        if self.refractive:
            return tf.math.imag(invert_update), tf.math.real(invert_update)
        else:
            return tf.math.abs(invert_update), tf.math.angle(invert_update)

    def _apply_object_update(
        self,
        objects: tf.Tensor,
        dexit: tf.Tensor,
        pre_exit: tf.Tensor,
        probe: tf.Tensor,
        prob_tf_abs: tf.Tensor,
        iter: int = 0,
    ) -> tf.Tensor:
        """Compute gradient from exit wave difference and apply object update.

        Handles all two probe modes (single_c, multi_c) including optional CNN corrections.

        Args:
            objects: Current complex object estimate.
            dexit: Exit wave difference to compute gradient from.
            pre_exit: Pre-constraint exit wave (needed for refractive gradient).
            probe: Probe function tensor.
            prob_tf_abs: Probe normalization factor.

        Returns:
            Updated complex object tensor.
        """
        mode = "refractive" if self.refractive else "polar"

        # single_c or multi_c probe mode
        invert_update = self.compute_gradient(dexit, pre_exit, probe, prob_tf_abs, reduce_modes=True)
        up_a, up_p = self.decompose_gradient(invert_update)
        update_a = self.cnn_tb_a(up_a * self.mask if self.mask is not None else up_a)
        update_p = self.cnn_tb_p(up_p * self.mask if self.mask is not None else up_p)
        update = combine_complex(update_a, update_p, mode=mode)
        return tf.cast(self.alpha[iter], "complex64") * update + objects

    def update_object(
        self,
        objects: tf.Tensor,
        exitw: tf.Tensor,
        pre_exit: tf.Tensor,
        probe: tf.Tensor,
        prob_tf_abs: tf.Tensor,
        iter: int = 0,
    ) -> tf.Tensor:
        """PIE update: object correction from direct exit wave difference."""
        dexit = exitw - pre_exit
        return self._apply_object_update(objects, dexit, pre_exit, probe, prob_tf_abs, iter=iter)

    def compute_output_intensity(self, objects: tf.Tensor, probe: tf.Tensor, fftconst: int) -> tf.Tensor:
        """Compute final diffraction amplitude from the refined object.

        Delegates to the module-level `ptychography_forward` function so
        the same logic can be reused outside of the refinement loop.
        """
        return ptychography_forward(objects, probe, fftconst, self.probe_mode, self.refractive)
