"""Total-variation regularisers on the reconstructed object volume.

Three flavours, all applied to a temporal stack ``[B, T, H, W]``:
- :func:`total_var` — 2D anisotropic TV per frame (no temporal term).
- :func:`total_var_3d` — 3D anisotropic TV with separate spatial and
  temporal terms.
- :func:`total_var_3d_iso` — 3D mixed TV: isotropic spatial gradient
  magnitude + L1 temporal gradient.  Default regulariser used by the
  :class:`~pid3net.layers.physics_layers.TV` layer.
"""

import tensorflow as tf


def total_var(images):
    """2D anisotropic total variation for a 4D tensor ``[B, T, H, W]``.

    No temporal coupling — TV is summed per-frame and averaged over T.
    """
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    sum_axis = [2, 3]
    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    scale = tf.cast(tf.shape(images)[-1] ** 2, "float32")
    time = tf.cast(tf.shape(images)[1], "float32")

    return tf.reduce_sum(total_vars) / (time * 2 * scale)


def total_var_3d(images):
    """3D anisotropic total variation for ``[B, T, H, W]`` with temporal regularisation.

    Spatial TV is L1 on `dx` and `dy`; temporal TV is L1 on `dt` with
    weight 0.5.  Both terms are scale-normalised.
    """
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    pixel_dif3 = images[:, 1:, :, :] - images[:, :-1, :, :]

    sum_axis = [2, 3]

    total_vars = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)
    total_vars_2 = tf.reduce_sum(tf.abs(pixel_dif3), axis=sum_axis)

    scale = tf.cast(tf.shape(images)[-1] ** 2, "float32")
    time = tf.cast(tf.shape(images)[1], "float32")

    return tf.reduce_sum(total_vars) / (time * 2 * scale) + 0.5 * tf.reduce_sum(total_vars_2) / ((time - 1) * scale)


def total_var_3d_iso(images):
    """3D mixed TV: isotropic spatial gradient magnitude + L1 temporal gradient.

    Spatial gradients use the isotropic norm ``sqrt(dx^2 + dy^2)``;
    temporal uses L1.  Default TV regulariser used by the ``TV`` layer.
    """
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    pixel_dif3 = images[:, 1:, :, :] - images[:, :-1, :, :]

    pixel_dif1 = tf.pad(pixel_dif1, [[0, 0], [0, 0], [0, 1], [0, 0]])
    pixel_dif2 = tf.pad(pixel_dif2, [[0, 0], [0, 0], [0, 0], [0, 1]])

    scale = tf.cast(tf.shape(images)[-1], "float32")
    total_vars = tf.reduce_sum(tf.sqrt(tf.abs(pixel_dif1) ** 2 + tf.abs(pixel_dif2) ** 2 + 1e-6))

    return total_vars / scale + tf.reduce_sum(tf.abs(pixel_dif3)) / scale**2
