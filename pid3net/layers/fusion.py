import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Activation


class PriorPhaseFusion(tf.keras.layers.Layer):
    """Fuse decoder phase/amplitude output with a per-step ODE-generated prior.

    The blending weight ``w`` is learned per-pixel per-step:
    - ``w → 0``: trust the decoder estimate.
    - ``w → 1``: trust the ODE prior.

    Args:
        filters: Number of intermediate Conv3D feature maps. Default 32.
        kernel_size: Spatial kernel size. Default 3.
        act: Activation for intermediate layers. Default ``"swish"``.
        **kwargs: Passed to ``tf.keras.layers.Layer``.

    Call signature:
        ``layer(x, prior)``

        x     : decoder phase or amplitude  [B, T, H, W]
        prior : ODE-generated prior         [B, T, H, W]  (same shape)

    Returns:
        Fused tensor of shape [B, T, H, W].
    """

    def __init__(self, filters: int = 32, kernel_size: int = 3, act: str = "swish", **kwargs: object) -> None:
        super().__init__(**kwargs)
        k = kernel_size
        self.convs = tf.keras.Sequential(
            [
                Conv3D(filters, (1, k, k), activation=act, padding="same"),
                Conv3D(filters, (1, k, k), activation=act, padding="same"),
                Conv3D(1, (1, k, k), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x: tf.Tensor, prior: tf.Tensor) -> tf.Tensor:
        feat = tf.stack([x, prior, x - prior], axis=-1)  # [B, T, H, W, 3]
        w = tf.squeeze(self.convs(feat), axis=-1)  # [B, T, H, W]
        return (1.0 - w) * x + w * prior


class PriorPhaseLoss(tf.keras.layers.Layer):
    """Weighted MSE loss between reconstruction and prior phase (or amplitude).

    Adds ``weight * mean((x - prior)²)`` as a regularisation loss via
    ``add_loss()``.  The input tensor ``x`` is returned unchanged (pass-through).

    The ``weight`` variable is **not** trainable — it is updated externally by
    the `PriorLossDecay` callback which
    cosine-anneals from a high initial value (trust prior early) to a low final
    value (trust network later).

    Args:
        weight: Initial loss weight (default 0.5).
        **kwargs: Passed to ``tf.keras.layers.Layer``.

    Call signature:
        ``layer(x, prior)``

        x     : reconstructed phase or amplitude  [B, T, H, W]
        prior : ODE-generated prior                [B, T, H, W]  (same shape)

    Returns:
        ``x`` unchanged.
    """

    def __init__(self, weight: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.weight = tf.Variable(weight, trainable=False, dtype="float32", name=f"lambda_{self.name}")

    def call(self, x: tf.Tensor, prior: tf.Tensor) -> tf.Tensor:
        self.add_loss(self.weight * tf.reduce_mean(tf.square(x - prior)))
        return x


class TimeDecayFusion(tf.keras.layers.Layer):
    def __init__(
        self,
        alpha: float = 0.1,
        filters: int = 8,
        kernel_size: int = 3,
        act: str = "swish",
        mode: str = "exp",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = tf.Variable(alpha, trainable=True, dtype=tf.float32)
        self.mode = mode
        self.convs = tf.keras.Sequential(
            [
                Conv3D(filters, (1, kernel_size, kernel_size), activation=act, padding="same"),
                Conv3D(filters, (1, kernel_size, kernel_size), activation=act, padding="same"),
                Conv3D(1, (1, kernel_size, kernel_size), activation="sigmoid", padding="same"),
            ]
        )
        self.act_out = Activation("sigmoid")

    def call(self, ap: tf.Tensor, init_ap: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(ap)
        init_ap = tf.tile(init_ap, [shape[0], shape[1], 1, 1])
        feat_imag = tf.stack([ap, init_ap, ap - init_ap], axis=-1)
        data_w = tf.squeeze(self.act_out(self.convs(feat_imag)), -1)

        is_first = tf.reshape(tf.cast(tf.equal(t, 0.0), tf.float32) * 0.1, (-1, 1, 1, 1))
        data_w = is_first + (1.0 - is_first) * data_w

        fused = (1 - data_w) * ap + data_w * init_ap
        return fused
