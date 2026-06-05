"""Reusable Conv blocks for the encoder/decoder backbones.

Two families:

- **Temporal blocks** (``Conv_Down_Temporal_Block``, ``Conv_Up_Temporal_Block``)
  operate on 5D tensors ``[B, T, H, W, C]`` and combine three temporal
  kernels (size 1, 3, 5 along the time axis) so the network sees multiple
  temporal receptive fields at once.  Used by 3D models.
- **2D blocks** (``Conv_Down_block``, ``Conv_Up_block``) operate on 4D
  tensors ``[B, H, W, C]`` and are simple two-conv units used by the 2D
  baseline.

All four use L2-regularised conv weights (1e-5) and offer either pooling
(downsampling) or transpose-conv / upsampling (upsampling).
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Conv2DTranspose,
    Conv3D,
    MaxPool3D,
    Conv3DTranspose,
    UpSampling3D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


class Conv_Down_Temporal_Block(keras.layers.Layer):
    """Multi-scale temporal conv block with optional spatial downsampling.

    Applies three parallel temporal convs (kernel sizes 1, 3, 5 along the
    time axis), concatenates them, projects back to ``nfilters`` channels,
    and optionally downsamples spatially.  Input/output are 5D temporal
    tensors ``[B, T, H, W, C]``.

    Args:
        nfilters: Number of output channels.
        w: Spatial kernel size for the temporal convs (default 3).
        p: Spatial pool size when ``pool`` is set (default 2).
        padding: Conv padding (default ``"same"``).
        pool: ``"max"`` → ``MaxPool3D``, ``"stride"`` → strided conv,
            otherwise no spatial downsampling.
        act: Activation function (default ``"swish"``).
        name: Layer name.
    """

    def __init__(
        self,
        nfilters: int,
        w: int = 3,
        p: int = 2,
        padding: str = "same",
        pool: object = None,
        act: str = "swish",
        name: str = "",
        **kwargs: object,
    ) -> None:
        super(Conv_Down_Temporal_Block, self).__init__(name=name, **kwargs)

        self.cv = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t1 = Conv3D(nfilters, (1, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t2 = Conv3D(nfilters // 2, (3, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t3 = Conv3D(nfilters // 2, (5, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_combine = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.norm = BatchNormalization()

        if pool == "max":
            self.pool = MaxPool3D((1, p, p), padding=padding)
        elif pool == "stride":
            self.pool = Conv3D(nfilters, (1, p, p), strides=(1, p, p), padding="valid")
        else:
            self.pool = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cv(x)
        x1 = self.cv_t1(x)
        x2 = self.cv_t2(x)
        x3 = self.cv_t3(x)
        x4 = tf.concat([x1, x2, x3], -1)
        x4 = self.cv_combine(x4)
        x4 = self.norm(x4)
        if self.pool is not None:
            x4 = self.pool(x4)
        return x4


class Conv_Up_Temporal_Block(keras.layers.Layer):
    """Multi-scale temporal conv block with 2× spatial upsampling.

    Mirror of `Conv_Down_Temporal_Block` but always upsamples spatially
    by 2 (via ``Conv3DTranspose`` if ``trans=True``, else ``UpSampling3D``).

    Args:
        nfilters: Number of output channels.
        w: Spatial kernel size (default 3).
        padding: Conv padding (default ``"same"``).
        trans: True → transposed-conv upsampling (learnable); False → nearest.
        act: Activation function (default ``"swish"``).
        name: Layer name.
    """

    def __init__(
        self,
        nfilters: int,
        w: int = 3,
        padding: str = "same",
        trans: bool = True,
        act: str = "swish",
        name: str = "",
        **kwargs: object,
    ) -> None:
        super(Conv_Up_Temporal_Block, self).__init__(name=name, **kwargs)

        self.cv = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t1 = Conv3D(nfilters, (1, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t2 = Conv3D(nfilters // 2, (3, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_t3 = Conv3D(nfilters // 2, (5, w, w), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv_combine = Conv3D(nfilters, (1, 1, 1), padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.norm = BatchNormalization()

        if trans:
            self.tcv = Conv3DTranspose(nfilters, (1, w, w), strides=(1, 2, 2), padding=padding)
        else:
            self.tcv = UpSampling3D(size=(1, 2, 2))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cv(x)
        x1 = self.cv_t1(x)
        x2 = self.cv_t2(x)
        x3 = self.cv_t3(x)
        x4 = tf.concat([x1, x2, x3], -1)
        x4 = self.cv_combine(x4)
        x4 = self.norm(x4)
        x4 = self.tcv(x4)
        return x4


class Conv_Down_block(keras.layers.Layer):
    """2D conv-conv block with optional spatial downsampling.

    Two stacked ``Conv2D`` layers + optional pooling.  Used by 2D models.

    Args:
        nfilters: Number of output channels.
        w: Kernel size (default 3).
        p: Pool size (default 2).
        padding: Conv padding (default ``"same"``).
        pool: ``"max"`` → ``MaxPool2D``, ``"stride"`` → strided conv, else none.
        act: Activation function (default ``"swish"``).
    """

    def __init__(
        self,
        nfilters: int,
        w: int = 3,
        p: int = 2,
        padding: str = "same",
        pool: object = None,
        act: str = "swish",
        **kwargs: object,
    ) -> None:
        super(Conv_Down_block, self).__init__(**kwargs)

        self.cv1 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv2 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))

        if pool == "max":
            self.pool = MaxPool2D(p, padding=padding)
        elif pool == "stride":
            self.pool = Conv2D(nfilters, w, 2, padding="same", activation=act)
        else:
            self.pool = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cv1(x)
        x = self.cv2(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Conv_Up_block(keras.layers.Layer):
    """2D conv-conv block with 2× transposed-conv spatial upsampling.

    Args:
        nfilters: Number of output channels.
        w: Kernel size (default 3).
        padding: Conv padding (default ``"same"``).
        act: Activation function (default ``"swish"``).
        trans: Kept for API parity with the temporal block; always transposed
            upsampling here.
    """

    def __init__(
        self,
        nfilters: int,
        w: int = 3,
        padding: str = "same",
        act: str = "swish",
        trans: bool = True,
        **kwargs: object,
    ) -> None:
        super(Conv_Up_block, self).__init__(**kwargs)

        self.cv1 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.cv2 = Conv2D(nfilters, w, padding=padding, activation=act, kernel_regularizer=l2(1e-5))
        self.tcv = Conv2DTranspose(nfilters, w, strides=2, padding=padding, kernel_regularizer=l2(1e-5))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.tcv(x)
        return x
