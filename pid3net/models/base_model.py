"""Base model class providing shared training, inference, and monitoring logic.

All PID3Net model variants inherit from PtyBase, which handles dataset loading,
training loop with cosine LR decay, checkpoint saving, reconstruction monitoring,
and batch inference.
"""

import os
import time

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay

from pid3net.utils.datagenerator_ssp import DataIteratorSsp
from pid3net.utils.general import dataset_functions
from pid3net.losses import negative_log_loss, masked_SEloss


class LearningRateTracker(tf.keras.callbacks.Callback):
    """Callback that prints the current learning rate at each epoch end."""

    def on_epoch_end(self, e, log):
        optimizer = self.model.optimizer
        print("\nLR: {:.6f}\n".format(optimizer._decayed_lr(tf.float32)))


class PriorLossDecay(tf.keras.callbacks.Callback):
    """Cosine-anneal the prior loss weight from initial to final over training.

    Works with :class:`~pid3net.layers.fusion.PriorPhaseLoss` layers whose
    ``weight`` variable controls the MSE loss strength between the
    reconstruction and the ODE phase/amplitude prior.

    Annealing schedule (cosine):
        ``w(epoch) = final + (initial - final) * (1 + cos(π * epoch / total)) / 2``

    Args:
        layer_names: List of PriorPhaseLoss layer names to update.
        initial: Starting loss weight (high = trust prior).
        final: Ending loss weight (low = trust network).
        total_epochs: Total number of training epochs.
    """

    def __init__(self, layer_names, initial, final, total_epochs):
        super().__init__()
        self.layer_names = layer_names
        self.initial = initial
        self.final = final
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / max(self.total_epochs, 1)
        new_weight = self.final + (self.initial - self.final) * (1 + np.cos(np.pi * progress)) / 2
        for name in self.layer_names:
            layer = self.model.get_layer(name)
            layer.weight.assign(new_weight)
        print(f"\nPrior loss weight: {new_weight:.4f}")


class ReconMonitor(tf.keras.callbacks.Callback):
    """Callback that saves phase/amplitude reconstruction visualizations each epoch.

    Generates a 2-row grid (phase top, amplitude bottom) for each temporal frame
    and saves as PNG to {save_path}/monitor/epoch_XXXX.png.
    """

    def __init__(self, trainIter, save_path, config):
        super().__init__()
        self.trainIter = trainIter
        self.save_path = save_path
        self.config = config
        os.makedirs(os.path.join(save_path, "monitor"), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        batch_x, _ = self.trainIter[0]
        diff = batch_x["diff"][:1]
        cfgh = self.config["hyper"]

        # Assemble model inputs respecting all optional branches.
        # init_pty and use_prior_phase can both be active simultaneously.
        model_inputs = [diff]
        if cfgh.get("init_pty", False):
            time_input = batch_x["time"][:1].reshape(-1, 1)
            model_inputs.append(time_input)
        if cfgh.get("use_prior_phase", False):
            if cfgh.get("use_prior_amp", False):
                model_inputs.append(batch_x["prior_amp"][:1])  # prior_amp first
            model_inputs.append(batch_x["prior_phase"][:1])  # then prior_phase

        outputs = self.model(model_inputs, training=False)
        _, a, p = outputs[0], outputs[1], outputs[2]

        p = np.array(p)[0]
        a = np.array(a)[0]

        n = p.shape[0]
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes[:, None]

        for i in range(n):
            im_p = axes[0, i].imshow(p[i], cmap="gray")
            axes[0, i].set_title(f"phase t={i}")
            axes[0, i].axis("off")
            plt.colorbar(im_p, ax=axes[0, i], fraction=0.046, pad=0.04)

            im_a = axes[1, i].imshow(a[i], cmap="RdBu_r")
            axes[1, i].set_title(f"amp t={i}")
            axes[1, i].axis("off")
            plt.colorbar(im_a, ax=axes[1, i], fraction=0.046, pad=0.04)

        fig.suptitle(f"Reconstruction — Epoch {epoch + 1}")
        plt.tight_layout()

        out_path = os.path.join(self.save_path, "monitor", f"epoch_{epoch + 1:04d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)


class PtyBase:
    """Base class for all ptychographic reconstruction models.

    Provides shared functionality for dataset loading, training with cosine LR decay,
    model checkpointing, and batch inference. Subclasses provide the Keras model
    via create_model().

    Class attribute:
        is_temporal: ``True`` if the model expects a temporal axis ``T`` in
            inputs (3D models); ``False`` for 2D models.  Controls batch shape,
            padding spec, ``n_time`` selection, and the inference loop.
            Default is ``True``; override in 2D subclasses.

    Args:
        config: Full config dict with 'model' and 'hyper' sections.
        model: Compiled tf.keras.Model instance.
        pretrained: Path to pretrained weights (handled by subclasses).
    """

    is_temporal: bool = True

    def __init__(self, config, model, pretrained=""):
        self.config = config
        self.model = model

    def create_dataset(self):
        """Load experimental data and create the training data iterator.

        Populates self.data_exp (raw data array) and self.trainIter (DataIteratorSsp).
        For 2D models, n_time is set to 1; for 3D models, uses config n_time.

        Passes optional config keys to the iterator:
        - ``phase_dir``, ``phase_file_pattern``, ``phase_key``, ``diff_dt_ms``,
          ``phase_dt_ms``: enable per-step ODE prior phase loading.
        """
        cfgh = self.config["hyper"]
        self.data_exp = dataset_functions[cfgh["sample"]](self.config)

        n_time = cfgh["n_time"] if self.is_temporal else 1

        self.trainIter = DataIteratorSsp(
            self.data_exp,
            batch_size=cfgh["batch_size"],
            image_size=self.config["model"]["img_size"],
            n_time=n_time,
            phase_dir=cfgh.get("phase_dir", None),
            phase_file_pattern=cfgh.get("phase_file_pattern", "f{time:04d}.npz"),
            use_prior_amp=cfgh.get("use_prior_amp", False),
            phase_key=cfgh.get("phase_key", "ap"),
            diff_dt_ms=cfgh.get("diff_dt_ms", 1.0),
            phase_dt_ms=cfgh.get("phase_dt_ms", 1.0),
        )

    def create_callbacks(self, epochs=1):
        """Create training callbacks: checkpoint, LR tracker, reconstruction monitor.

        Args:
            epochs: Total training epochs (needed for PriorLossDecay annealing).
        """
        save_path = self.config["hyper"]["save_path"]
        cfgh = self.config["hyper"]

        callbacks = [
            ModelCheckpoint(
                filepath=f"{save_path}/models/model_unsp.tf",
                monitor="loss",
                save_weights_only=True,
                verbose=2,
                save_best_only=True,
            ),
            LearningRateTracker(),
            ReconMonitor(self.trainIter, save_path, self.config),
        ]

        # Cosine-anneal prior loss weight: high early (trust prior), low later
        if cfgh.get("use_prior_phase", False):
            layer_names = ["prior_phase_loss"]
            if cfgh.get("use_prior_amp", False):
                layer_names.append("prior_amp_loss")
            callbacks.append(
                PriorLossDecay(
                    layer_names=layer_names,
                    initial=cfgh.get("lambda_prior", 10.0),
                    final=cfgh.get("lambda_prior_min", 1.0),
                    total_epochs=epochs,
                )
            )

        return callbacks

    def create_loss_op(self):
        """Select loss function based on config: Poisson NLL (dist=True) or masked MSE."""
        if self.config["hyper"]["dist"]:
            return negative_log_loss(self.config["hyper"]["loss"])
        return masked_SEloss

    def train(self, epochs):
        """Run the training loop.

        Compiles the model with Adam optimizer and cosine decay LR schedule.
        Saves model checkpoints (best loss), config snapshot, and per-epoch
        reconstruction visualizations.

        Args:
            epochs: Number of training epochs.

        Returns:
            tf.keras.callbacks.History object with training metrics.
        """
        lr_schedule = CosineDecay(
            self.config["hyper"]["lr"],
            1.0 * epochs * len(self.trainIter),
            alpha=0.2,
        )
        loss = self.create_loss_op()
        cfgh = self.config["hyper"]

        self.model.compile(loss=[loss, None, None], optimizer=tf.keras.optimizers.Adam(lr_schedule))

        save_path = self.config["hyper"]["save_path"]
        os.makedirs(f"{save_path}/models/", exist_ok=True)

        callbacks = self.create_callbacks(epochs=epochs)

        with open(f"{save_path}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        # Disable multiprocessing when the data generator does file I/O (phase_dir
        # set).  Forked worker processes inherit TF's CUDA state from the parent but
        # cannot re-initialise it, producing CUDA_ERROR_NOT_INITIALIZED errors.
        # Without file I/O the original multi-worker setup is safe to keep.
        has_file_io = bool(cfgh.get("phase_dir", None))
        self.hist = self.model.fit(
            self.trainIter,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
            use_multiprocessing=not has_file_io,
            workers=1 if has_file_io else 4,
        )
        return self.hist

    def get_padding_info(self):
        """Compute padding dimensions when img_size > data spatial size.

        Returns:
            Tuple of (padding_amount, pad_spec) where pad_spec is a numpy pad
            specification or None if no padding needed.
        """
        padding = self.config["model"]["img_size"] - self.data_exp.shape[-1]
        pad_spec = None
        if padding > 0:
            half = padding // 2
            if self.is_temporal:
                pad_spec = ((0, 0), (0, 0), (half, half), (half, half))
            else:
                pad_spec = ((0, 0), (half, half), (half, half))
        return padding, pad_spec

    def prepare_batch(self, data, padding, pad_spec):
        """Prepare a data batch for model input: add batch dim (3D) and apply padding."""
        if self.is_temporal:
            data = data[None, ...]
        if padding > 0:
            data = np.pad(data, pad_spec)
        return data

    def get_batch_size(self):
        """Return the inference batch size: ``n_time`` for 3D models, ``batch_size`` for 2D."""
        cfgh = self.config["hyper"]
        return cfgh["n_time"] if self.is_temporal else cfgh["batch_size"]

    def _build_model_inputs(self, diff_batch, start_frame_idx: int):
        """Assemble the model input list for inference.

        Handles all combinations of optional inputs:
        - ``init_pty``: appends a ``time`` index array shaped ``[1, T, 1]``.
        - ``use_prior_phase``: loads the ODE-generated prior phase from disk via
          the data iterator's phase-loading helpers, pads it to ``img_size``, and
          appends it shaped ``[1, T, H, W]``.

        Args:
            diff_batch: Prepared diffraction array ``[1, T, H, W]`` (already padded).
            start_frame_idx: Index of the first frame in the current batch within
                ``self.data_exp`` — used to align the prior-phase file lookup.

        Returns:
            List of input arrays/tensors ready to pass to ``self.model``.
        """
        cfgh = self.config["hyper"]
        inputs = [diff_batch]

        if cfgh.get("init_pty", False):
            size = diff_batch.shape[1]  # T dimension
            time_input = np.arange(start_frame_idx, start_frame_idx + size).reshape(-1, 1)
            inputs.append(time_input)

        if cfgh.get("use_prior_phase", False):
            # Load both amplitude and phase priors from the same files.
            # _load_prior_sequence returns {"amp": [1,T,H,W], "phase": [1,T,H,W]}.
            size = diff_batch.shape[1]  # T
            h = w = diff_batch.shape[-1]  # H/W after padding
            frame_indexes = [start_frame_idx]
            prior = self.trainIter._load_prior_sequence(
                frame_indexes,
                diff_shape=(1, size, h, w),
            )
            if cfgh.get("use_prior_amp", False):
                inputs.append(prior["amp"])  # prior_amp
            inputs.append(prior["phase"])  # prior_phase

        return inputs

    def inference(self, reload=False):
        """Run inference on the full dataset and save reconstructed amplitude and phase.

        Processes the dataset in chunks of batch_size (2D) or n_time (3D).
        Saves results to {save_path}/object_reconstruction_{model}.npz containing
        [amplitude_array, phase_array] each of shape (N_frames, H, W).
        """
        if reload:
            self.model.load_weights(f"{self.config['hyper']['save_path']}/models/model_unsp.tf").expect_partial()

        all_predict_a = []
        all_predict_p = []

        t = time.time()
        padding, pad_spec = self.get_padding_info()
        size = self.get_batch_size()

        for idx in range(0, len(self.data_exp) // size, 1):
            diff = self.data_exp[idx * size : (idx + 1) * size]
            diff = self.prepare_batch(diff, padding, pad_spec)
            inputs = self._build_model_inputs(diff, start_frame_idx=idx * size)
            outputs = self.model(inputs, training=False)
            _, a, p = outputs[0], outputs[1], outputs[2]

            all_predict_a.append(a.numpy())
            all_predict_p.append(p.numpy())

        print("Total Inferences time: ", time.time() - t)

        all_predict_a = np.concatenate(all_predict_a, axis=0).reshape(-1, a.shape[-1], a.shape[-1])
        all_predict_p = np.concatenate(all_predict_p, axis=0).reshape(-1, a.shape[-1], a.shape[-1])

        save_path = self.config["hyper"]["save_path"]
        model = self.config["model"]["model"]
        np.savez_compressed(f"{save_path}/object_reconstruction_{model}.npz", [all_predict_a, all_predict_p])

    def inference_o(self, overlap=4):
        """Run inference with overlapping temporal windows for smoother temporal transitions.

        Loads the best checkpoint and processes the dataset with sliding windows
        that overlap by the specified number of frames.

        Args:
            overlap: Number of overlapping frames between consecutive windows.
        """
        self.model.load_weights(f"{self.config['hyper']['save_path']}/models/model_unsp.tf").expect_partial()

        all_predict_a = []
        all_predict_p = []

        t = time.time()
        padding, pad_spec = self.get_padding_info()
        size = self.get_batch_size()

        diff = self.data_exp[0:size]
        diff = self.prepare_batch(diff, padding, pad_spec)
        inputs = self._build_model_inputs(diff, start_frame_idx=0)
        outputs = self.model(inputs)
        _, a, p = outputs[0], outputs[1], outputs[2]
        all_predict_a.append(a[0])
        all_predict_p.append(p[0])

        for idx in range(size - overlap, len(self.data_exp) - size + 1, size - overlap):
            diff = self.data_exp[idx : idx + size]
            diff = self.prepare_batch(diff, padding, pad_spec)
            inputs = self._build_model_inputs(diff, start_frame_idx=idx)
            outputs = self.model(inputs)
            _, a, p = outputs[0], outputs[1], outputs[2]
            all_predict_a.append(a[0])
            all_predict_p.append(p[0])

        print("Total Inferences time: ", time.time() - t)

        all_predict_a = np.array(all_predict_a).reshape(-1, a.shape[-1], a.shape[-1])
        all_predict_p = np.array(all_predict_p).reshape(-1, a.shape[-1], a.shape[-1])

        save_path = self.config["hyper"]["save_path"]
        model = self.config["model"]["model"]
        np.savez_compressed(
            f"{save_path}/object_reconstruction_{model}_overlap_{overlap}.npz",
            [all_predict_a, all_predict_p],
        )
