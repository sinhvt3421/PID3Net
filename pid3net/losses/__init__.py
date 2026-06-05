"""Loss functions and regularisers for ptychographic reconstruction.

Public surface (backward-compatible with the old flat
``pid3net/losses.py``):

- Diffraction training losses — see :mod:`pid3net.losses.diffraction`:
    :func:`negative_log_loss`, :func:`negative_log_loss_t`,
    :func:`masked_SEloss`, :func:`log10` (helper).
- Object-space TV regularisers — see :mod:`pid3net.losses.regularizers`:
    :func:`total_var`, :func:`total_var_3d`, :func:`total_var_3d_iso`.

Existing imports of the form
``from pid3net.losses import total_var_3d, negative_log_loss, ...``
continue to work unchanged.
"""

from pid3net.losses.diffraction import (
    log10,
    masked_SEloss,
    negative_log_loss,
    negative_log_loss_t,
)
from pid3net.losses.regularizers import (
    total_var,
    total_var_3d,
    total_var_3d_iso,
)

__all__ = [
    "log10",
    "masked_SEloss",
    "negative_log_loss",
    "negative_log_loss_t",
    "total_var",
    "total_var_3d",
    "total_var_3d_iso",
]
