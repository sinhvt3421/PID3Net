# Adding a model

PID3Net dispatches models through `pid3net.models.MODEL_REGISTRY`.
Adding a new variant takes **one file plus one registry entry** — no
edits to `base_model.py` or `train.py`.

## The contract

A model class must:

1. Inherit from `pid3net.models.base_model.PtyBase`.
2. Declare `is_temporal: bool` as a **class attribute** (read by `PtyBase`
   to branch between 3D and 2D dataset handling).
3. Accept `(config, pretrained="")` in its `__init__`.
4. Construct a compiled `tf.keras.Model` (typically via a helper
   `create_model(config)` function in the same module) and forward
   it to `super().__init__(config=config, model=model)`.

## Minimal example

Create `pid3net/models/my_model.py`:

```python
"""MyModel — a 3D temporal variant with a tweaked decoder."""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from pid3net.models.base_model import PtyBase
from pid3net.layers.encoders import TBEncoder
from pid3net.layers.decoders import TBDecoder
from pid3net.layers.physics_layers import RefineLayer, CombineComplex


class MyModel(PtyBase):
    """3D temporal model with whatever distinguishing feature."""

    is_temporal = True  # required class attribute

    def __init__(self, config, pretrained=""):
        model = create_model(config)
        if pretrained:
            model.load_weights(pretrained).expect_partial()
        super().__init__(config=config, model=model)


def create_model(config):
    cfgh = config["hyper"]
    cfgm = config["model"]
    # ... build a Keras model: Input → TBEncoder → TBDecoder
    # → CombineComplex → RefineLayer → output_head
    # See pid3net/models/pid3net.py for the full pattern.
    ...
    return Model(inputs=[...], outputs=[...])
```

## Register it

Two ways:

=== "Import-time registration (recommended for in-tree models)"

    Edit `pid3net/models/registry.py`:

    ```python
    from pid3net.models.my_model import MyModel

    MODEL_REGISTRY: Dict[str, ModelSpec] = {
        # ... existing entries ...
        "mymodel": ModelSpec(name="MyModel", cls=MyModel),
    }
    ```

    Also re-export from `pid3net/models/__init__.py` for tidiness.

=== "Runtime registration (out-of-tree extensions)"

    From a user script:

    ```python
    from pid3net.models import register_model
    from my_package import MyModel

    register_model("mymodel", "MyModel", MyModel)
    ```

    Then `pid3net-train --model mymodel ...` works.

## Use it

```bash
pid3net-train configs/my_experiment.yaml --model mymodel
```

Or in YAML:

```yaml
model:
  model: "mymodel"
```

## Why a class attribute, not a constructor argument?

`PtyBase` reads `self.is_temporal` in several places to branch on data
dimensionality (e.g. building 3D vs 2D input tensors, picking 3D vs 2D
loss reductions). Putting the flag on the class makes the contract
explicit at the type level — `cls.is_temporal` works without
instantiating, which the registry uses for shape introspection.

## See also

- [Architecture](../concepts/architecture.md) — what existing models
  look like.
- [`pid3net.models`](../api/models.md) — full registry API.
- [`pid3net.models.PtyBase`](../api/models.md#pid3net.models.PtyBase) —
  base class.
