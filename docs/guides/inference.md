# Inference

Inference re-runs the model on the training data using saved weights,
producing the reconstructed object stack without further training.

## Run-only mode

```bash
pid3net-train configs/my_experiment.yaml --inference-only
```

This:

1. Loads the YAML config.
2. Instantiates the model with the same architecture as training.
3. Loads weights from the auto-suffixed `save_path` (or from
   `--pretrained` if supplied).
4. Forward-passes the dataset and writes the output stack.

## Specify a checkpoint explicitly

```bash
pid3net-train configs/my_experiment.yaml \
    --inference-only \
    --pretrained trained_models/<run-name>/models/model_unsp.tf
```

Useful when comparing checkpoints from different runs.

## Output structure

Inference writes `object_reconstruction.npz` (alongside the existing
training artifacts):

```python
import numpy as np
out = np.load("trained_models/<run-name>/object_reconstruction.npz")
print(out.files)
# ['amplitude', 'phase', 'diff_intensity']
amp   = out["amplitude"]       # shape [T, H, W]
phase = out["phase"]           # shape [T, H, W]
diff  = out["diff_intensity"]  # shape [T, H, W], predicted intensity
```

The exact key set is determined by the model's output head — see
`pid3net.models.pid3net.build_output_head`.

## Throughput and memory

A single forward pass through `n_refine = 5` refinement iterations
on `[T, 256, 256]` runs in ~50 ms per frame on a single A100 GPU.
For longer stacks, the data generator yields batches of size 1 by
default — adjust `batch_size` in the YAML's `hyper:` block for
parallel inference if you have the memory.

Inference uses the same per-batch FFT path as training, so peak GPU
memory is dominated by the refinement-loop intermediate tensors
(`probe_size² · T · n_refine` complex values).

## See also

- [Training](training.md) — for context on how the weights got saved.
- [Configuration](configuration.md) — `batch_size`, `save_path`.
