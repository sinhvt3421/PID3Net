"""Loss functions and regularisers for ptychographic reconstruction.

Public surface (backward-compatible with the old flat
``pid3net/losses.py``):

- Diffraction training losses — see `diffraction`:
    `negative_log_loss`, `negative_log_loss_t`,
    `masked_SEloss`, `log10` (helper).
- Object-space TV regularisers — see `regularizers`:
    `total_var`, `total_var_3d`, `total_var_3d_iso`.

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
