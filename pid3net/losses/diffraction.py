"""Output-space training losses for the diffraction prediction head.

Two flavours:
- :func:`negative_log_loss` — Poisson NLL on
    the predicted intensity distribution (used with ``--dist``).
- :func:`masked_SEloss` — masked squared error on sqrt-intensity (default).

All losses ignore very-low-intensity pixels (below ``min_val``), which on
Poisson data are dominated by quantisation/photon noise and would otherwise
add a structured noise floor to the gradient.
"""

import tensorflow as tf


def log10(x):
    """Base-10 logarithm — used to scale Poisson NLL by the per-batch intensity range."""
    return tf.math.log(x) / tf.math.log(10.0)


def negative_log_loss(min_val=1.0):
    """Poisson NLL with a hard reliability mask on pixels above ``min_val``.

    Differs from :func:`negative_log_loss_t` by summing only over reliable
    pixels (mask × log-prob) and reducing over the spatial axes explicitly.
    """

    def nll(y_true, y_pred):
        norm = tf.math.floor(log10(tf.reduce_max(y_true)))
        # Poisson SNR: sqrt(I). Weight by reliability.
        mask = tf.cast(y_true > min_val, tf.float32)

        pixel_loss = -y_pred.log_prob(y_true) * mask
        return tf.reduce_sum(pixel_loss, axis=[-1, -2]) / tf.pow(10.0, norm)

    return nll


def masked_SEloss(y_true, y_pred):
    """Masked squared error on sqrt-intensity, ignoring zero-valued pixels.

    Computes ``(sqrt(y_pred) - sqrt(y_true))^2`` only where ``y_true != 0``.

    Args:
        y_true: Ground truth diffraction intensity.
        y_pred: Predicted diffraction intensity.

    Returns:
        Scalar loss value.
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    masked_squared_error = tf.square(tf.sqrt(y_pred) - tf.sqrt(y_true))
    masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask)
    return masked_mse
