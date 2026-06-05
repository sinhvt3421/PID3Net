"""Encoder backbones built from temporal or 2D conv blocks.

- `TBEncoder` — temporal block encoder for 3D models
  (input ``[B, T, H, W, C]``).
- `CNNEncoder` — 2D encoder for non-temporal models
  (input ``[B, H, W, C]``).

Both produce a "latent" feature map at the bottom of the pyramid; spatial
size is reduced by a factor of ``k_pool ** n_layers`` while channels grow
geometrically as ``filters * 2**i``.
"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers

from pid3net.layers.conv_blocks import Conv_Down_Temporal_Block, Conv_Down_block


class TBEncoder(keras.layers.Layer):
    """Temporal-block encoder for 3D models.

    Stack of ``n_layers`` `Conv_Down_Temporal_Block` layers with growing
    channel counts (``filters * 2**i``) followed by one non-pooling latent
    block.

    Args:
        n_layers: Number of downsampling blocks.
        filters: Base channel count of the first block.
        w: Spatial kernel size.
        k_pool: Spatial pool factor at each downsampling block.
        pool: ``"max"`` / ``"stride"`` / None (see `Conv_Down_Temporal_Block`).
        activation: Activation function.
        name: Layer name.
    """

    def __init__(
        self,
        n_layers: int = 4,
        filters: int = 8,
        w: int = 3,
        k_pool: int = 2,
        pool: str = "max",
        activation: str = "swish",
        name: str = "",
        **kwargs: object,
    ) -> None:
        super(TBEncoder, self).__init__(name=name, **kwargs)

        self.tb_down = [
            Conv_Down_Temporal_Block(
                filters * 2**i, w, k_pool, pool=pool, act=activation, name="tb_encoder_{}".format(i)
            )
            for i in range(n_layers)
        ]

        self.latent = Conv_Down_Temporal_Block(
            filters * 2 ** (n_layers - 1), w, act=activation, pool=None, name="latent"
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for i in range(len(self.tb_down)):
            x = self.tb_down[i](x)
        x = self.latent(x)
        return x


class CNNEncoder(keras.layers.Layer):
    """2D-conv encoder for non-temporal models.

    2D counterpart of `TBEncoder`.  Stacks ``n_layers``
    `Conv_Down_block` layers + one latent block.  Args match
    `TBEncoder`.
    """

    def __init__(
        self,
        n_layers: int = 4,
        filters: int = 8,
        w: int = 3,
        k_pool: int = 2,
        pool: str = "max",
        activation: str = "swish",
        name: str = "",
        **kwargs: object,
    ) -> None:
        super(CNNEncoder, self).__init__(name=name, **kwargs)

        self.cnn_down = [
            Conv_Down_block(filters * 2**i, w, k_pool, pool=pool, act=activation, name="encoder_{}".format(i))
            for i in range(n_layers)
        ]

        self.latent = Conv_Down_block(filters * 2 ** (n_layers - 1), w, act=activation, pool=None, name="latent")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for i in range(len(self.cnn_down)):
            x = self.cnn_down[i](x)
        x = self.latent(x)
        return x
