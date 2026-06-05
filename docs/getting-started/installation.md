# Installation

## Requirements

| | |
|---|---|
| Python | ≥ 3.9 (tested through 3.11) |
| TensorFlow | `>=2.10, <2.13` |
| TensorFlow-Probability | `>=0.18, <0.20` |
| CUDA (optional) | follow the [TF GPU guide](https://www.tensorflow.org/install) |

GPU is optional but strongly recommended for training. On a CPU,
inference works but training is slow.

## Create an environment

```bash
conda create -n pid3net python=3.10
conda activate pid3net
```

For CUDA support, install the GPU build of TensorFlow before the
`pid3net` install:

```bash
conda install -c conda-forge tensorflow-gpu
```

## Install PID3Net

=== "Editable (recommended for development)"

    ```bash
    git clone https://github.com/sinhvt3421/PID3Net
    cd PID3Net
    pip install -e ".[dev]"
    ```

    The `[dev]` extra adds `pytest`, `ruff`, and `black`. Drop it for
    a runtime-only install.

=== "From source (wheel build)"

    ```bash
    git clone https://github.com/sinhvt3421/PID3Net
    cd PID3Net
    pip install build
    python -m build
    pip install dist/pid3net-*.whl
    ```

=== "With docs extras"

    ```bash
    pip install -e ".[docs]"
    mkdocs serve
    ```

## Verify the install

```bash
python -c "import pid3net; print(pid3net.__version__)"
# 2.0.0

python -c "from pid3net.models import PID3Net, MODEL_REGISTRY; print(MODEL_REGISTRY['3d3'])"
# ModelSpec(name='PID3Net', cls=<class 'pid3net.models.pid3net.PID3Net'>)

pid3net-train --help
```

If `pid3net-train` is on your `PATH`, the install is complete.

## Optional extras

| Extra | Adds |
|---|---|
| `dev` | `pytest`, `pytest-datadir`, `pytest-benchmark`, `ruff`, `black` |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `pymdown-extensions`, `mkdocs-include-markdown-plugin` |

Combine with `pip install -e ".[dev,docs]"`.

## Next steps

- [Quickstart](quickstart.md) — train the default model on the
  bundled moving-chart config.
- [Apply to a new dataset](new-dataset.md) — point PID3Net at your own
  measurements.
