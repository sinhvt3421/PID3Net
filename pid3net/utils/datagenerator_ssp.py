import os
import logging
from math import ceil
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.image import resize

RNG_SEED = 2134
logger = logging.getLogger(__name__)


class DataIteratorSsp(Sequence):
    """Data iterator for self-supervised ptychography training.

    Returns diffraction batches with an optional per-step prior phase loaded
    from pre-computed files (e.g. ODE-interpolated phases from a high-exposure
    PID3Net reconstruction).

    When ``phase_dir`` is ``None`` the iterator behaves identically to the
    original v4 implementation — no phase files are loaded and the output
    dict contains only ``{"diff", "time"}``.

    When ``phase_dir`` is provided the output dict gains a ``"prior_phase"``
    key carrying an array of shape ``[B, T, H, W]`` aligned to the
    corresponding diffraction frames.

    Time alignment
    --------------
    For diffraction frame index ``t_diff`` the aligned phase frame is::

        t_phase = floor(t_diff * diff_dt_ms / phase_dt_ms)

    When the ODE trajectory is already frame-aligned to the low-exposure
    diffraction (the common case) set both ``diff_dt_ms`` and
    ``phase_dt_ms`` to ``1.0`` (the defaults).

    Phase file format
    -----------------
    Each file stores an array of shape ``(N, H, W, ≥2)`` (``npz``) or
    ``(H, W, ≥2)`` (``npy``), where N is the number of retrieval
    variations.  Channel 0 = amplitude, channel 1 = phase.  One variation
    is chosen randomly per load.

    Args:
        data: Diffraction amplitude array of shape ``[N_frames, H, W]``.
        batch_size: Number of temporal sequences per batch.
        n_time: Number of consecutive frames per sequence (temporal window).
        image_size: Target spatial size; data is zero-padded if smaller.
        phase_dir: Directory containing per-frame phase files.  ``None``
            disables phase loading (default).
        phase_file_pattern: ``str.format`` pattern for phase filenames.
            Must include ``{time}`` and end with ``.npy`` or ``.npz``.
            Default ``"f{time:04d}.npz"``.
        phase_key: Key used when loading ``.npz`` files. Default ``"xhat"``.
        diff_dt_ms: Time step of diffraction frames in ms. Default ``1.0``.
        phase_dt_ms: Time step of phase files in ms. Default ``1.0``.
        use_prior_amp: When True (and ``phase_dir`` is set), include
            ``"prior_amp"`` in the input dict (channel 0 of prior files).
            Default False — amplitude priors degrade quality for weakly
            absorbing samples.
        seed: Random seed for reproducible variation selection and index
            shuffling.  ``None`` for non-deterministic behaviour.
    """

    def __init__(
        self,
        data: "np.ndarray",
        batch_size: int = 16,
        n_time: int = 5,
        image_size: int = 256,
        phase_dir: "str | None" = None,
        phase_file_pattern: str = "f{time:04d}.npz",
        phase_key: str = "xhat",
        diff_dt_ms: float = 1.0,
        phase_dt_ms: float = 1.0,
        use_prior_amp: bool = False,
        seed: "int | None" = None,
    ) -> None:
        self.use_prior_amp = use_prior_amp
        self.batch_size = batch_size
        self.data = data
        self.n_time = n_time
        self.data_indexes = range(len(self.data) - self.n_time + 1)
        self.image_size = image_size
        self.padding = self.image_size - self.data.shape[-1]

        # Optional phase loading
        self.phase_dir = phase_dir
        self.phase_file_pattern = phase_file_pattern
        self.phase_key = phase_key
        self.diff_dt_ms = float(diff_dt_ms)
        self.phase_dt_ms = float(phase_dt_ms)

        self.rng = np.random.default_rng(seed)

        self.on_epoch_end()

    # ------------------------------------------------------------------
    # Keras Sequence bookkeeping
    # ------------------------------------------------------------------

    def on_epoch_end(self):
        self.indexes = self.rng.choice(
            list(self.data_indexes),
            len(self.data) // self.n_time,
            replace=False,
        )

    def __len__(self):
        return ceil(len(self.data) / (self.batch_size * self.n_time))

    # ------------------------------------------------------------------
    # Phase loading helpers (only used when phase_dir is set)
    # ------------------------------------------------------------------

    def _map_diff_time_to_phase_time(self, t_diff: int) -> int:
        """Map diffraction frame index to phase file index (floor alignment)."""
        t_phys_ms = t_diff * self.diff_dt_ms
        t_phase = int(np.floor(t_phys_ms / self.phase_dt_ms))
        return max(t_phase, 0)

    def _load_prior_for_diff_time(self, t_diff: int, target_hw) -> dict:
        """Load amplitude and phase priors aligned to diffraction frame ``t_diff``.

        File layout (channel-last):
            ``.npy``  → ``(H, W, 2)``  — single frame, channel 0 = amp, 1 = phase
            ``.npz``  → ``(N, H, W, 2)``  — N retrieval variations; one is sampled

        Returns:
            dict with keys ``"amp"`` and ``"phase"``, each ``(H, W)`` float32.
        """
        t_phase = self._map_diff_time_to_phase_time(t_diff)
        fpath = os.path.join(
            self.phase_dir,
            self.phase_file_pattern.format(time=t_phase),
        )

        if self.phase_file_pattern.endswith(".npy"):
            arr = np.load(fpath)  # (H, W, 2)
            if arr.ndim != 3 or arr.shape[-1] < 2:
                raise ValueError(f"Expected .npy prior array of shape (H, W, ≥2), " f"got {arr.shape} in {fpath}")
            arr = arr[np.newaxis]  # → (1, H, W, C)

        elif self.phase_file_pattern.endswith(".npz"):
            arr = np.load(fpath)[self.phase_key]  # (N, H, W, 2)
            if arr.ndim != 3 or arr.shape[-1] < 2:
                raise ValueError(f"Expected .npy prior array of shape (H, W, ≥2), " f"got {arr.shape} in {fpath}")
            arr = arr[np.newaxis]  # → (1, H, W, C)

        else:
            raise ValueError(
                f"Unknown file pattern '{self.phase_file_pattern}'. " "Use a pattern ending in '.npy' or '.npz'."
            )

        n_var = arr.shape[0]
        if n_var < 1:
            raise ValueError(f"No prior entries found in {fpath}")

        vidx = int(self.rng.integers(0, n_var))
        amp = arr[vidx, ..., 0].astype(np.float32)  # (H, W)
        phase = arr[vidx, ..., 1].astype(np.float32)  # (H, W)

        h_tgt, w_tgt = target_hw
        if amp.shape[0] != h_tgt or amp.shape[1] != w_tgt:
            amp = resize(amp[..., np.newaxis], (h_tgt, w_tgt), method="bilinear").numpy()[..., 0]
            phase = resize(phase[..., np.newaxis], (h_tgt, w_tgt), method="bilinear").numpy()[..., 0]

        return {"amp": amp.astype(np.float32), "phase": phase.astype(np.float32)}

    def _load_prior_sequence(self, indexes, diff_shape) -> dict:
        """Build per-step prior amplitude and phase arrays aligned to ``indexes``.

        Returns:
            dict with keys ``"amp"`` and ``"phase"``, each of shape
            ``[B, T, H, W]`` when ``n_time > 1`` or ``[B, H, W]`` when
            ``n_time == 1``.
        """
        h, w = diff_shape[-2], diff_shape[-1]

        if self.n_time > 1:
            amp_seqs, phase_seqs = [], []
            for k in indexes:
                frames = [self._load_prior_for_diff_time(t, target_hw=(h, w)) for t in range(k, k + self.n_time)]
                amp_seqs.append(np.stack([f["amp"] for f in frames], axis=0))  # (T,H,W)
                phase_seqs.append(np.stack([f["phase"] for f in frames], axis=0))
            return {
                "amp": np.stack(amp_seqs, axis=0).astype(np.float32),  # (B,T,H,W)
                "phase": np.stack(phase_seqs, axis=0).astype(np.float32),
            }
        else:
            frames = [self._load_prior_for_diff_time(k, target_hw=(h, w)) for k in indexes]
            return {
                "amp": np.stack([f["amp"] for f in frames], axis=0).astype(np.float32),  # (B,H,W)
                "phase": np.stack([f["phase"] for f in frames], axis=0).astype(np.float32),
            }

    # ------------------------------------------------------------------
    # Spatial padding helper
    # ------------------------------------------------------------------

    def _pad_spatial(self, x: np.ndarray) -> np.ndarray:
        """Symmetrically zero-pad the H and W dimensions of ``x``.

        Supports arrays with any number of leading batch/time dimensions.
        The last two axes are treated as H and W respectively.
        """
        if self.padding <= 0:
            return x
        p1 = self.padding // 2
        p2 = self.padding - p1
        pad = [(0, 0)] * x.ndim
        pad[-2] = (p1, p2)  # H
        pad[-1] = (p1, p2)  # W
        return np.pad(x, pad, mode="constant")

    # ------------------------------------------------------------------
    # Main item getter
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.n_time > 1:
            diff = np.array(
                [self.data[k : k + self.n_time] for k in indexes],
                dtype=np.float32,
            )
        else:
            diff = np.array(self.data[indexes], dtype=np.float32)

        diff_p = self._pad_spatial(diff)

        inputs = {"diff": diff_p, "time": indexes}

        if self.phase_dir is not None:
            prior = self._load_prior_sequence(indexes, diff.shape)
            if self.use_prior_amp:
                inputs["prior_amp"] = self._pad_spatial(prior["amp"])
            inputs["prior_phase"] = self._pad_spatial(prior["phase"])

        y_main = diff**2  # [B, T, H, W] — per-frame measured intensity
        return inputs, y_main
