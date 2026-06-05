"""Model classes and the central model registry.

Use `MODEL_REGISTRY` (or `register_model`) to discover or add
model variants without editing ``train_ssp.py`` or ``base_model.py``.
"""

from pid3net.models.base_model import PtyBase
from pid3net.models.baseline import PIBaseD3Net
from pid3net.models.pid3net import PID3Net
from pid3net.models.registry import MODEL_REGISTRY, ModelSpec, get_spec, register_model

__all__ = [
    "PtyBase",
    "PIBaseD3Net",
    "PID3Net",
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_spec",
    "register_model",
]
