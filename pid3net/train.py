"""Training and inference entry point for PID3Net reconstruction models.

Supports the 3D temporal ``PID3Net`` and its 2D ablation ``PIBaseD3Net``,
configured via YAML files and CLI arguments. See README.md for usage examples.

Invocation:
- As an installed console script: ``pid3net-train <dataset.yaml> [options]``
- As a module: ``python -m pid3net.train <dataset.yaml> [options]``
- Legacy compat: ``python train_ssp.py <dataset.yaml> [options]`` (root shim).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf

from pid3net.models import MODEL_REGISTRY, get_spec

logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU memory growth to avoid pre-allocating all VRAM."""
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info("Using GPU: %s", physical_devices[0].name)
        except RuntimeError as e:
            logger.warning("GPU setup failed: %s", e)
    else:
        logger.info("No GPU found, running on CPU")


def set_seed(seed=0):
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(args):
    """Load YAML config and apply CLI argument overrides.

    Order of precedence (later wins):
    1. YAML file contents
    2. Standard CLI args (``--model``, ``--n_refine``, ``--probe_mode``, ...)

    The ``save_path`` is auto-suffixed with model, loss type, refinement
    steps, probe mode, update method, and seed so multiple runs don't
    collide.
    """
    with open(args.dataset) as f:
        config = yaml.safe_load(f)

    config["hyper"]["dist"] = args.dist
    config["hyper"]["n_refine"] = args.n_refine
    config["hyper"]["probe_mode"] = args.probe_mode
    config["hyper"]["rec_mode"] = args.rec_mode
    config["hyper"]["update_method"] = args.update_method

    config["model"]["model"] = args.model

    config["hyper"].setdefault("tvo", False)

    config["hyper"]["save_path"] += "_{}_{}_{}_r{}_{}_{}_{}_seed{}".format(
        args.model,
        "poiss" if args.dist else "mse",
        config["hyper"]["loss"],
        args.n_refine,
        args.probe_mode,
        args.rec_mode,
        args.update_method,
        args.seed,
    )

    return config


def load_model(args, config):
    """Instantiate a model from the registry based on ``--model``."""
    spec = get_spec(config)
    logger.info("Loading model: %s, reconstruction mode: %s", spec.name, config["hyper"]["rec_mode"])
    return spec.cls(config, args.pretrained)


def _archive_sources(save_path):
    """Copy the model / physics / loss source files next to the checkpoint.

    Uses package-relative ``__file__`` paths so the archive works whether
    the package is installed editable, from a wheel, or run from a source
    checkout.  Missing files / directories are silently skipped (e.g. when
    the package is run from a stripped wheel without source).
    """
    pkg_root = Path(__file__).resolve().parent
    file_sources = [
        pkg_root / "models" / "pid3net.py",
        pkg_root / "models" / "base_model.py",
        pkg_root / "layers" / "physics_layers.py",
    ]
    dir_sources = [
        pkg_root / "losses",
    ]
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    for src in file_sources:
        if src.exists():
            shutil.copy(src, save_dir / src.name)
    for src in dir_sources:
        if src.exists():
            shutil.copytree(src, save_dir / src.name, dirs_exist_ok=True)


def run(args):
    """Run the full training and/or inference pipeline.

    Pipeline: GPU setup -> seed -> load config -> build model -> load dataset
    -> train (unless ``--inference-only``) -> archive sources -> inference.
    """
    setup_gpu()
    set_seed(args.seed)

    config = load_config(args)
    ssp_model = load_model(args, config)
    ssp_model.model.summary()

    logger.info("Loading dataset and creating training set")
    ssp_model.create_dataset()

    if not args.inference_only:
        logger.info("Starting training for %d epochs", args.epoch)
        start = time.time()
        hist = ssp_model.train(args.epoch)
        logger.info("Total training time: %.1fs", time.time() - start)
        np.save(config["hyper"]["save_path"] + "/hist_train.npy", hist.history)

    _archive_sources(config["hyper"]["save_path"])

    logger.info("Running inference")
    ssp_model.inference(reload=not args.inference_only)


def build_parser():
    """Build the argparse parser used by both the console script and the shim."""
    parser = argparse.ArgumentParser(description="Train PID3Net models for dynamic CXDI reconstruction")
    parser.add_argument("dataset", type=str, help="Path to dataset YAML config")
    parser.add_argument(
        "--model", type=str, default="3d3", choices=list(MODEL_REGISTRY.keys()), help="Model architecture to use"
    )
    parser.add_argument("--n_refine", type=int, default=5, help="Refinement steps for reconstruction")
    parser.add_argument(
        "--probe_mode", type=str, default="multi_c", help="Probe mode: single_c, multi_c"
    )
    parser.add_argument("--rec_mode", type=str, default="refractive", help="Reconstruction mode polar or refractive")
    parser.add_argument(
        "--update_method",
        type=str,
        default="pie",
        choices=["pie", "raar"],
        help="Refinement update method: pie (ePIE, default) or raar (Relaxed Averaged Alternating Reflections)",
    )
    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained model weights")
    parser.add_argument("--dist", action="store_true", help="Use Poisson distribution output")
    parser.add_argument("--epoch", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--inference-only", action="store_true", help="Skip training, run inference only")
    return parser


def main(argv: "list[str] | None" = None) -> None:
    """Console-script entry point.

    Args:
        argv: Optional list of argument strings.  When ``None`` (the default)
            arguments are read from ``sys.argv``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
