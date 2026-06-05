"""PID3Net model: 3D temporal encoder-decoder with physics-informed refinement.

Architecture: log-transform input -> TBEncoder -> dual TBDecoder (amplitude + phase)
-> optional masking/time-decay fusion -> CombineComplex (refractive mode)
-> RefineLayer (iterative FFT-based refinement) -> output head.

The model operates in refractive index mode where the complex object is encoded as
phi + j*amp (real part = phase, imaginary part = amplitude).
"""

import tensorflow as tf
import numpy as np

import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Conv3D

from pid3net.models.base_model import PtyBase
from pid3net.layers.activations import Mpi, PhaseConstraint
from pid3net.layers.physics_layers import (
    CombineComplex,
    TV,
    RefineLayer,
)
from pid3net.layers.fusion import TimeDecayFusion, PriorPhaseFusion, PriorPhaseLoss
from pid3net.layers.encoders import TBEncoder
from pid3net.layers.decoders import TBDecoder

tfpl = tfp.layers
tfd = tfp.distributions


class PID3Net(PtyBase):
    """Main PID3Net model with refractive refinement and temporal fusion.

    Args:
        config: Full config dict with 'model' and 'hyper' sections.
        pretrained: Path to pretrained weights file. Empty string to skip.
    """

    def __init__(self, config: dict, pretrained: str = "") -> None:
        model = create_model(config)
        if pretrained:
            print("Load pretrained model from ", pretrained)
            model.load_weights(pretrained).expect_partial()
        super().__init__(config=config, model=model)


def load_physics_data(cfgh):
    """Load probe function, spatial mask, and initial reconstruction from disk.

    Args:
        cfgh: Hyperparameter config dict.

    Returns:
        Tuple of (probs, mask, object_init) where mask and object_init are None if disabled.
        object_init is a tuple (amplitude_init, phase_init) each of shape [1, 1, H, W].
    """
    probs = np.load(cfgh["probe"], allow_pickle=True)
    if cfgh["probe_norm"]:
        probs = tf.constant(probs * np.sqrt(float(cfgh["probe_norm"])), dtype="complex64")

    mask = None
    if cfgh["masking"]:
        mask = np.load(cfgh["masking"], allow_pickle=True)[None, ...]
        mask = tf.constant(mask, dtype="float32")

    object_init = None
    if cfgh["init_pty"]:
        raw = np.load(cfgh["init_pty"], allow_pickle=True)
        object_init = (
            tf.constant(raw[0], dtype="float32")[None, None, ...],
            tf.constant(raw[1], dtype="float32")[None, None, ...],
        )
        print("Using initial ptycho from ", cfgh["init_pty"])

    return probs, mask, object_init


def build_inputs(cfgm, cfgh):
    """Create Keras Input layers for diffraction data and optional auxiliary inputs.

    Args:
        cfgm: Model config dict.
        cfgh: Hyperparameter config dict.

    Returns:
        Tuple of (diff_input, time_input, prior_amp_input, prior_phase_input).
        ``time_input`` is None if ``init_pty`` is disabled.
        ``prior_phase_input`` is None if ``use_prior_phase`` is False/absent.
        ``prior_amp_input`` is None unless both ``use_prior_phase`` and
        ``use_prior_amp`` are True.
    """
    diff = Input(name="diff", shape=(None, cfgm["img_size"], cfgm["img_size"], 1), dtype="float32")
    time_input = Input(name="time", shape=(1,), dtype="float32") if cfgh["init_pty"] else None

    _prior_shape = (None, cfgm["img_size"], cfgm["img_size"])
    _use_prior = cfgh.get("use_prior_phase", False)
    prior_amp_input = (
        Input(name="prior_amp", shape=_prior_shape, dtype="float32")
        if _use_prior and cfgh.get("use_prior_amp", False)
        else None
    )
    prior_phase_input = Input(name="prior_phase", shape=_prior_shape, dtype="float32") if _use_prior else None
    return diff, time_input, prior_amp_input, prior_phase_input


def build_encoder_decoder(diff, cfgm, rec_mode="polar"):
    """Build the encoder-decoder backbone for amplitude and phase estimation.

    Applies log-transform to input, encodes with TBEncoder, then decodes with
    two separate TBDecoders for amplitude and phase branches.

    In polar mode, a PhaseConstraint (tanh-based) is applied to bound phase to
    [-alpha, alpha].  In refractive mode the constraint is skipped so the phase
    output is unbounded, avoiding tanh saturation that causes phase collapse.

    Args:
        diff: Input diffraction tensor [B, T, H, W, 1].
        cfgm: Model config dict with encoder/decoder parameters.
        rec_mode: Reconstruction mode ('polar' or 'refractive').

    Returns:
        Tuple of (amplitude, phase) tensors, each [B, T, H, W].
    """
    e = tf.math.log(diff + 1e-9)

    latent = TBEncoder(
        n_layers=cfgm["n_cov"],
        filters=cfgm["filters"],
        w=cfgm["kernel"],
        k_pool=cfgm["k_pool"],
        pool=cfgm["pool"],
        name="encoder_tb",
    )(e)

    da = TBDecoder(n_layers=cfgm["n_dcov"], filters=cfgm["filters"], w=cfgm["kernel"], name="decoder_amp")(latent)
    a = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid" if rec_mode == "polar" else None)(da)

    a = Lambda(lambda x: tf.squeeze(x, -1), name="amp")(a)

    dp = TBDecoder(n_layers=cfgm["n_dcov"], filters=cfgm["filters"], w=cfgm["kernel"], name="decoder_phase")(latent)
    p = Conv3D(1, (1, 1, 1), padding="same", activation=None)(dp)

    if rec_mode == "polar":
        p = Mpi(name="phase_constraint")(p)
    else:
        p = PhaseConstraint(name="phase_constraint")(p)

    p = Lambda(lambda x: tf.squeeze(x, -1), name="phi")(p)

    return a, p


def crop_to_probe_size(a, p, diff, probs):
    """Crop decoded outputs to match probe spatial dimensions.

    When img_size > probe_size, the encoder-decoder operates on padded input.
    This removes the padding from amplitude, phase, and diffraction tensors.

    Args:
        a: Amplitude tensor [B, T, H, W].
        p: Phase tensor [B, T, H, W].
        diff: Input diffraction tensor [B, T, H, W, 1].
        probs: Probe array (used for its spatial shape).

    Returns:
        Tuple of (a_cropped, p_cropped, diff_cropped) matching probe dimensions.
    """
    padding = diff.shape[-2] - probs.shape[-1]

    if padding == 0:
        diff_pad = diff[..., 0]
    else:
        h = padding // 2
        diff_pad = diff[:, :, h:-h, h:-h, 0]
        a = a[:, :, h:-h, h:-h]
        p = p[:, :, h:-h, h:-h]

    return a, p, diff_pad


def apply_masking_and_ptycho_fusion(a, p, mask, object_init, time_input, cfgh):
    """Apply spatial masking and time-decay fusion with initial reconstruction.

    If masking is enabled, phase is multiplied by the spatial mask.
    If init_pty is enabled, amplitude and phase are blended with the initial
    reconstruction using learned time-dependent weights via TimeDecayFusion.

    Args:
        a: Amplitude tensor.
        p: Phase tensor.
        mask: Spatial mask tensor or None.
        object_init: Tuple (amp_init, phase_init) or None.
        time_input: Time index tensor or None.
        cfgh: Hyperparameter config dict.

    Returns:
        Tuple of (amplitude, phase) after masking and fusion.
    """
    if cfgh["masking"]:
        a = Lambda(lambda x: x, name="amplitude")(a)
        p = Lambda(lambda x: x * mask, name="phase")(p)

    if cfgh["init_pty"] and object_init is not None:
        object_init_a, object_init_p = object_init
        a = TimeDecayFusion(name="fuse_a")(a, object_init_a, time_input)
        p = TimeDecayFusion(name="fuse_p")(p, object_init_p, time_input)

    return a, p


def apply_prior_fusion(a, p, prior_amp_input, prior_phase_input, cfgh):
    """Fuse decoder amplitude and phase outputs with ODE-generated per-step priors.

    When ``use_prior_phase=True``, applies `PriorPhaseFusion`
    independently to both branches:

    - **Amplitude branch**: blends ``a`` with ``prior_amp`` (channel 0 of prior file).
    - **Phase branch**: blends ``p`` with ``prior_phase`` (channel 1 of prior file).

    Both use a per-pixel learnable weight ``w ∈ [0,1]``:
    ``out = (1 − w) * decoder + w * prior``.

    Args:
        a: Decoder amplitude tensor ``[B, T, H, W]``.
        p: Decoder phase tensor ``[B, T, H, W]``.
        prior_amp_input: Keras Input for ODE amplitude prior ``[B, T, H, W]`` or None.
        prior_phase_input: Keras Input for ODE phase prior ``[B, T, H, W]`` or None.
        cfgh: Hyperparameter config dict (uses ``"use_prior_phase"`` key).

    Returns:
        Tuple ``(amplitude, phase)`` after fusion (unchanged if disabled).
    """
    if cfgh.get("use_prior_phase", False):
        if prior_amp_input is not None:
            a = PriorPhaseFusion(name="prior_amp_fusion")(a, prior_amp_input)
        if prior_phase_input is not None:
            p = PriorPhaseFusion(name="prior_phase_fusion")(p, prior_phase_input)
    return a, p


def crop_prior_to_probe_size(prior_input, diff, probs):
    """Center-crop a prior tensor from img_size down to probe_size.

    After `crop_to_probe_size` the amplitude/phase tensors are at probe_size
    ``[H_p, W_p]``.  The prior was built at img_size, so it must be cropped by the
    same amount before it can be used for loss computation or fusion.

    Args:
        prior_input: Keras tensor ``[B, T, img_size, img_size]`` or None.
        diff: Original diffraction Input tensor ``[B, T, img_size, img_size, 1]``
            — used only to read the spatial size via ``diff.shape[-2]``.
        probs: Probe array — ``probs.shape[-1]`` gives probe_size.

    Returns:
        Cropped tensor ``[B, T, probe_size, probe_size]`` or None if
        ``prior_input`` is None.
    """
    if prior_input is None:
        return None
    padding = diff.shape[-2] - probs.shape[-1]
    if padding == 0:
        return prior_input
    h = padding // 2
    return prior_input[:, :, h:-h, h:-h]


def combine_and_regularize(a, p, cfgh):
    """Combine amplitude and phase into complex object with TV regularization.

    When tvo=False: TV is applied separately on amplitude and phase before combining.
    When tvo=True: TV is applied on the combined complex object.

    Args:
        a: Amplitude tensor.
        p: Phase tensor.
        cfgh: Hyperparameter config dict (uses 'tvo' key).

    Returns:
        Complex object tensor in refractive mode (real=phase, imag=amplitude).
    """
    if not cfgh["tvo"]:
        a = TV(0.01, "tv_a")(a)
        p = TV(0.01, "tv_p")(p)

    objects = CombineComplex()(a, p, cfgh["rec_mode"])
    if cfgh["tvo"]:
        objects = TV(0.5, "tv_o")(objects)

    return objects


def apply_refinement(objects, diff_pad, probs, mask, cfgh):
    """Apply iterative physics-informed refinement using the RefineLayer.

    Performs n_refine iterations of: forward FFT -> intensity constraint -> inverse FFT
    -> object update. Uses refractive mode or polar mode.

    Args:
        objects: Complex object tensor to refine.
        diff_pad: Measured diffraction intensity (target).
        probs: Probe function array.
        mask: Spatial mask tensor or None.
        cfgh: Hyperparameter config dict.

    Returns:
        Tuple of ``(refined_diffraction_amplitude, refined_objects)``.
    """
    if "single" in cfgh["probe_mode"]:
        probe_lr = tf.constant(probs[None, ...], dtype="complex64")
    else:
        probe_lr = tf.constant(probs[:, None], dtype="complex64")

    refine = RefineLayer(
        mask if cfgh["masking"] else None,
        cfgh["n_refine"],
        cfgh["probe_mode"],
        refractive=cfgh["rec_mode"] == "refractive",
        update_method=cfgh.get("update_method", "pie"),
        refine_cfg=cfgh.get("refine", {}),
    )
    return refine(objects, diff_pad, probe_lr, probs.shape[-1])


def build_output_head(
    diff_amp_r,
    objects_r,
    mask,
    cfgh,
    prior_phase_cropped=None,
    prior_amp_cropped=None,
    probs=None,
):
    """Build the output layers: diffraction intensity, refined amplitude, refined phase.

    When ``use_prior_phase=True``, a `PriorPhaseLoss`
    layer adds a weighted MSE loss between ``phase_r`` and the ODE prior.  The
    weight is cosine-annealed by `PriorLossDecay`.

    Args:
        diff_amp_r: Refined diffraction amplitude from RefineLayer ``[B, T, H, W]``.
        objects_r: Refined complex object from RefineLayer.
        mask: Spatial mask tensor or None.
        cfgh: Hyperparameter config dict.
        prior_phase_cropped: ODE phase prior cropped to probe size, or None.
        prior_amp_cropped: ODE amplitude prior cropped to probe size, or None.

    Returns:
        List of output tensors ``[diff_intensity, amplitude_r, phase_r]`` (3 items),
    """
    if cfgh["tvo"]:
        objects_r = TV(1.0, "tv_or", True)(objects_r)

    if cfgh["rec_mode"] == "refractive":
        ar = Lambda(lambda x: tf.math.imag(x) * mask if cfgh["masking"] else tf.math.imag(x), name="amplitude_r")(
            objects_r
        )
        pr = Lambda(lambda x: tf.math.real(x) * mask if cfgh["masking"] else tf.math.real(x), name="phase_r")(objects_r)

    else:
        ar = Lambda(lambda x: tf.math.abs(x) * mask if cfgh["masking"] else tf.math.abs(x), name="amplitude_r")(
            objects_r
        )
        pr = Lambda(lambda x: tf.math.angle(x) * mask if cfgh["masking"] else tf.math.angle(x), name="phase_r")(
            objects_r
        )

    if not cfgh["tvo"]:
        ar = TV(0.01, "tv_ar", True)(ar)
        pr = TV(0.1, "tv_pr", True)(pr)

    # Prior loss: annealing MSE between refined output and ODE prior.
    # Weight is cosine-annealed by PriorLossDecay callback (high early, low later).
    if cfgh.get("use_prior_phase", False) and prior_phase_cropped is not None:
        pr = PriorPhaseLoss(weight=cfgh.get("lambda_prior", 10.0), name="prior_phase_loss")(pr, prior_phase_cropped)
    if cfgh.get("use_prior_amp", False) and prior_amp_cropped is not None:
        ar = PriorPhaseLoss(weight=cfgh.get("lambda_prior", 1.0), name="prior_amp_loss")(ar, prior_amp_cropped)

    if cfgh["dist"]:
        diff_out = tfpl.DistributionLambda(
            lambda x: tfd.Poisson(x**2),
            name="diff_intensity_poiss",
        )(diff_amp_r)
    else:
        diff_out = Lambda(lambda x: x**2, name="diff_intensity")(diff_amp_r)

    output = [diff_out, ar, pr]

    return output


def create_model(config):
    """Build the full PID3Net Keras model from config.

    Orchestrates all builder functions::

        load physics data
        → build inputs (diff, optional time, optional prior_phase)
        → encoder-decoder backbone
        → crop to probe size
        → spatial masking + TimeDecayFusion (if init_pty)
        → PriorPhaseFusion on phase branch (if use_prior_phase)
        → TV regularisation + CombineComplex
        → RefineLayer (physics-informed iterative updates)
        → output head (diff intensity, amplitude_r, phase_r)

    Args:
        config: Full config dict with ``"model"`` and ``"hyper"`` sections.

    Config keys (``hyper`` section):
        use_prior_phase (bool): If True, enables prior-phase guidance via
            `PriorPhaseFusion` (pre-refinement
            blending with ODE prior) and
            `PriorPhaseLoss` (annealing MSE loss
            on refined phase vs prior, weight decayed by
            `PriorLossDecay`).  Default False.
        use_prior_amp (bool): If True (and use_prior_phase is also True), additionally
            fuses the ODE amplitude prior with the amplitude decoder branch and
            adds an annealing amplitude prior loss.  Disabled by default because
            amplitude priors degrade weakly-absorbing samples.  Default False.
        lambda_prior (float): Initial prior loss weight (default 0.5).
        lambda_prior_min (float): Final prior loss weight after annealing (default 0.01).

    Returns:
        ``tf.keras.Model`` with outputs ``[diff_intensity, amplitude_r, phase_r]``
        Inputs (in order): ``[diff]``, then optionally ``time`` (if init_pty),
        then optionally ``prior_amp`` (if use_prior_amp), then optionally
        ``prior_phase`` (if use_prior_phase).
    """
    cfgm = config["model"]
    cfgh = config["hyper"]

    probs, mask, object_init = load_physics_data(cfgh)
    diff, time_input, prior_amp_input, prior_phase_input = build_inputs(cfgm, cfgh)

    a, p = build_encoder_decoder(diff, cfgm, rec_mode=cfgh["rec_mode"])

    a, p, diff_pad = crop_to_probe_size(a, p, diff, probs)

    # Crop priors to probe_size for the annealing loss in build_output_head
    prior_phase_cropped = crop_prior_to_probe_size(prior_phase_input, diff, probs)
    prior_amp_cropped = crop_prior_to_probe_size(prior_amp_input, diff, probs)

    # Pre-refinement fusion: blend both decoder branches with initial from ptychography
    a, p = apply_masking_and_ptycho_fusion(a, p, mask, object_init, time_input, cfgh)

    # Pre-refinement fusion: blend both decoder branches with generated/ODE per-step priors
    a, p = apply_prior_fusion(a, p, prior_amp_cropped, prior_phase_cropped, cfgh)

    objects = combine_and_regularize(a, p, cfgh)

    diff_amp_r, objects_r = apply_refinement(objects, diff_pad, probs, mask, cfgh)

    output = build_output_head(
        diff_amp_r,
        objects_r,
        mask,
        cfgh,
        prior_phase_cropped=prior_phase_cropped,
        prior_amp_cropped=prior_amp_cropped,
        probs=probs,
    )

    # Collect non-None inputs in a consistent order:
    # diff → (time) → (prior_amp if use_prior_amp) → (prior_phase if use_prior_phase)
    inputs = [diff]
    if cfgh["init_pty"]:
        inputs.append(time_input)
    if cfgh.get("use_prior_phase", False):
        if cfgh.get("use_prior_amp", False) and prior_amp_input is not None:
            inputs.append(prior_amp_input)
        inputs.append(prior_phase_input)

    return Model(inputs=inputs, outputs=output)
