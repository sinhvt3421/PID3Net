"""Per-dataset loaders for measured diffraction stacks.

Each loader reads the ``hyper.train_data`` path from the config, applies
dataset-specific preprocessing (centre-cropping, zero-clipping, rolling),
and returns the **amplitude** stack ``sqrt(intensity)`` of shape
``[N_frames, H, W]`` (float32) suitable as the diffraction input to the
model.

Loaders are dispatched via the ``dataset_functions`` registry, keyed by the
``hyper.sample`` config value.  Built-in keys: ``"chart"``, ``"simu"``,
``"aunp"``, ``"mgall"``.  Register additional loaders by adding entries.

All loaders expect ``train_data`` to be an ``.npz`` file whose first array
(``"arr_0"``) contains the measured diffraction intensity stack
``[N_frames, H, W]``.
"""

import numpy as np


def load_aunp_data(cfg):
    """Load AuNP diffraction data with off-centre crop to ``img_size``.

    Reads ``train_data`` ``[N, H, W]``, takes the first 1000 frames, crops
    around centre ``(250, 249)`` to ``img_size × img_size``, returns
    ``sqrt(intensity)``.

    Args:
        cfg: Full config dict (uses ``hyper.train_data`` and ``model.img_size``).

    Returns:
        Amplitude stack ``[1000, img_size, img_size]`` (float32).
    """
    data = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    size = cfg["model"]["img_size"]
    data = data[:1000, 250 - size // 2 : 250 + size // 2, 249 - size // 2 : 249 + size // 2]
    return np.sqrt(data)


def load_chart_data(cfg):
    """Load moving-chart diffraction data with a 1-pixel roll along axis 2.

    The roll corrects for a known half-pixel offset in the simulated chart
    data.  Returns ``sqrt(intensity)``.

    Args:
        cfg: Full config dict (uses ``hyper.train_data``).

    Returns:
        Amplitude stack ``[N, H, W]`` (float32).
    """
    data = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    data = np.roll(data, shift=1, axis=2)
    return np.sqrt(data)


def load_simu_data(cfg):
    """Load simulated diffraction data without preprocessing.

    Args:
        cfg: Full config dict (uses ``hyper.train_data``).

    Returns:
        Amplitude stack ``[N, H, W]`` (float32) = ``sqrt(intensity)``.
    """
    obs = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    return np.sqrt(obs)


def load_mg_data(cfg):
    """Load Mg-alloy diffraction data with negative-clip, roll, and centre crop.

    Pipeline: clip negative values to 0 → roll axis 2 by 1 pixel → centre-crop
    to ``img_size × img_size`` → ``sqrt(intensity)``.

    Args:
        cfg: Full config dict (uses ``hyper.train_data`` and ``model.img_size``).

    Returns:
        Amplitude stack ``[N, img_size, img_size]`` (float32).
    """
    data = np.load(cfg["hyper"]["train_data"], allow_pickle=True)["arr_0"]
    data[data < 0] = 0
    data = np.roll(data, shift=1, axis=2)

    size = cfg["model"]["img_size"]
    org_size = data.shape[1]
    obs = np.sqrt(
        data[
            :,
            org_size // 2 - size // 2 : org_size // 2 + size // 2,
            org_size // 2 - size // 2 : org_size // 2 + size // 2,
        ]
    )
    return obs


#: Registry mapping ``hyper.sample`` config keys to dataset loader functions.
#: Extend at import time by inserting new entries.
dataset_functions = {
    "chart": load_chart_data,
    "simu": load_simu_data,
    "aunp": load_aunp_data,
    "mgall": load_mg_data,
}
