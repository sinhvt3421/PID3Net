"""Decoder backbones built from temporal or 2D conv blocks.

- :class:`TBDecoder` — temporal block decoder for 3D models.
- :class:`CNNDecoder` — 2D decoder for non-temporal models.

Each decoder mirrors its encoder: ``n_layers`` upsampling blocks (channels
shrinking as ``filters * 2**(n_layers-1-i)``) plus a final 1×k×k projection
to ``filters`` channels.  Output channels are not collapsed to 1 — the
caller adds a final ``Conv3D(1, ...)`` or ``Conv2D(1, ...)`` per branch.
"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Lambda, Conv3D, MultiHeadAttention, LayerNormalization, Dense, Flatten

from pid3net.layers.conv_blocks import Conv_Up_Temporal_Block, Conv_Up_block


class TBDecoder(keras.layers.Layer):
    """Temporal-block decoder for 3D models.

    Stacks ``n_layers - 1`` :class:`Conv_Up_Temporal_Block` layers with
    shrinking channel counts, one final ``Conv_Up_Temporal_Block`` at
    ``filters`` channels, and a 1×w×w projection.

    Args:
        n_layers: Number of upsampling blocks (matches encoder depth).
        filters: Base channel count (matches encoder's ``filters``).
        w: Spatial kernel size.
        activation: Activation function.
        name: Layer name.
    """

    def __init__(self, n_layers=4, filters=8, w=3, activation="swish", name="", **kwargs):
        super(TBDecoder, self).__init__(name=name, **kwargs)

        self.tb_up = [
            Conv_Up_Temporal_Block(filters / 2 * 2 ** (n_layers - i), w, act=activation, name="tb_decoder_{}".format(i))
            for i in range(n_layers - 1)
        ]

        self.tb_up_last = Conv_Up_Temporal_Block(filters, w, name="decoder_{}".format(n_layers - 1))
        self.out = Conv3D(filters, (1, w, w), padding="same", activation="swish")

    def call(self, x):
        for i in range(len(self.tb_up)):
            x = self.tb_up[i](x)
        x = self.tb_up_last(x)
        x = self.out(x)
        return x


class CNNDecoder(keras.layers.Layer):
    """2D-conv decoder for non-temporal models.

    2D counterpart of :class:`TBDecoder`.  Args match.
    """

    def __init__(self, n_layers=4, filters=8, w=3, activation="swish", name="", **kwargs):
        super(CNNDecoder, self).__init__(name=name, **kwargs)

        self.tb_up = [
            Conv_Up_block(filters / 2 * 2 ** (n_layers - i), w, act=activation, name="decoder_{}".format(i))
            for i in range(n_layers - 1)
        ]

        self.tb_up_last = Conv_Up_block(filters, w, name="decoder_{}".format(n_layers - 1))
        self.out = Conv2D(filters, w, padding="same", activation="swish")

    def call(self, x):
        for i in range(len(self.tb_up)):
            x = self.tb_up[i](x)
        x = self.tb_up_last(x)
        x = self.out(x)
        return x
