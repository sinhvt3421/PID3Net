"""Central registry of model variants.

Each entry maps a short config key (e.g. ``"3d3"``) to a :class:`ModelSpec`
describing the model class and a human-readable name.  The model class itself
declares its ``is_temporal`` flag as a class attribute (read by
:class:`~pid3net.models.base_model.PtyBase`).

Adding a new model variant requires only:

1. Implement a new ``PtyBase`` subclass in ``pid3net/models/``.  Set
   ``is_temporal = True`` (3D) or ``False`` (2D) as a class attribute.
2. Register it here with :func:`register_model` (or extend ``MODEL_REGISTRY``).
3. Re-export the class from ``pid3net/models/__init__.py``.

No edits to ``base_model.py`` or ``pid3net/train.py`` are required.
"""

from dataclasses import dataclass
from typing import Dict, Type

from pid3net.models.base_model import PtyBase
from pid3net.models.baseline import PIBaseD3Net
from pid3net.models.pid3net import PID3Net


@dataclass(frozen=True)
class ModelSpec:
    """Describes a model variant for the training pipeline.

    Args:
        name: Human-readable name (used in logs).
        cls: The ``PtyBase`` subclass implementing the model.
    """

    name: str
    cls: Type[PtyBase]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "3d3": ModelSpec(name="PID3Net", cls=PID3Net),
    "2d": ModelSpec(name="PIBaseD3Net", cls=PIBaseD3Net),
}


def register_model(key: str, name: str, cls: Type[PtyBase]) -> None:
    """Register a new model variant at runtime.

    Args:
        key: Short config key (matches ``config["model"]["model"]``).
        name: Human-readable name shown in logs.
        cls: The ``PtyBase`` subclass implementing the model.  Must declare
            ``is_temporal = True|False`` as a class attribute.
    """
    MODEL_REGISTRY[key] = ModelSpec(name=name, cls=cls)


def get_spec(config) -> ModelSpec:
    """Return the ``ModelSpec`` for the model named in ``config["model"]["model"]``.

    Raises:
        ValueError: If the model key is not registered.
    """
    key = config["model"]["model"]
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]
