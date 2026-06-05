"""Bounded activation layers for amplitude and phase constraints.

These layers wrap a single scalar trainable variable around a saturating
nonlinearity (``tanh`` or ``clip``) so the network output is forced into a
physically meaningful range while the bound itself is learned during training.

Three variants exist for three different output domains:

- :class:`AmpConstraint`: hard-clipped amplitude (used for refractive-mode
  CNN amplitude updates where the value must stay in a small bounded range
  but is not naturally probabilistic).
- :class:`Mpi`: ``tanh(x) * alpha`` with ``alpha`` clipped to ``[-π, π]`` — the
  natural bound for **polar-mode** phase, which physically wraps modulo ``2π``.
- :class:`PhaseConstraint`: ``tanh(x) * alpha`` with ``alpha`` clipped to
  ``[-10, 10]`` — the wider bound used for **refractive-mode** phase, where the
  phase value can grow beyond ``π`` (no wrapping), so the constraint exists
  only to stabilise training, not to enforce a physical limit.

``Mpi`` and ``PhaseConstraint`` share the same formula but differ in their
initial alpha (0.5 vs 3.0/4.0) and clipping range — they are kept as separate
classes so each reconstruction mode can use the right default without runtime
mode-checking.
"""

import math

import tensorflow as tf


class AmpConstraint(tf.keras.layers.Layer):
    """Hard clip to ``[-0.5, 5.0]`` for amplitude in refractive-mode CNN updates.

    No trainable parameter — the bounds are fixed.  Used inside
    :class:`~pid3net.layers.physics_layers.CNNTBLayer` when ``out="const"``.

    Call:
        ``layer(x) -> tf.clip_by_value(x, -0.5, 5.0)``
    """

    def call(self, inputs):
        return tf.clip_by_value(inputs, -0.5, 5.0)


class PhaseConstraint(tf.keras.layers.Layer):
    """Bounded phase activation for **refractive-mode** reconstruction.

    Computes ``tanh(x) * alpha`` where ``alpha`` is a trainable scalar clipped
    to ``[-10, 10]``.  The wide range accommodates refractive-mode phase, which
    is not wrapped to ``[-π, π]`` and may grow beyond ``π`` for thick or
    strongly-phase-shifting samples.

    The constraint stabilises training (prevents runaway phase) without
    imposing the polar-mode ``±π`` bound.

    Args:
        **kwargs: Passed to ``tf.keras.layers.Layer``.

    Trainable:
        alpha (scalar, init 3.0, clipped to ``[-10, 10]``).
    """

    def __init__(self, **kwargs):
        super(PhaseConstraint, self).__init__(**kwargs)
        self.alpha = tf.Variable(
            3.0, name="alpha_act", trainable=True, constraint=lambda x: tf.clip_by_value(x, -10, 10)
        )

    def call(self, inputs):
        return tf.math.tanh(inputs) * self.alpha


class Mpi(tf.keras.layers.Layer):
    """Bounded phase activation for **polar-mode** reconstruction.

    Computes ``tanh(x) * alpha`` where ``alpha`` is a trainable scalar clipped
    to ``[-π, π]``.  The hard bound matches polar mode's physical phase range
    (object = ``amp * exp(j*phase)``), which is undefined outside one period.

    Args:
        **kwargs: Passed to ``tf.keras.layers.Layer``.

    Trainable:
        alpha (scalar, init 0.5, clipped to ``[-π, π]``).
    """

    def __init__(self, **kwargs):
        super(Mpi, self).__init__(**kwargs)
        self.alpha = tf.Variable(
            0.5, name="alpha_act", trainable=True, constraint=lambda x: tf.clip_by_value(x, -math.pi, math.pi)
        )

    def call(self, inputs):
        return tf.math.tanh(inputs) * self.alpha
