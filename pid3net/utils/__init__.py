"""Utility modules: data iterators and dataset loaders.

Public surface:
- ``DataIteratorSsp``: Keras Sequence data generator for self-supervised
  ptychographic training. Yields per-batch dicts of inputs (diffraction,
  optional time index, optional prior amp/phase) and per-batch targets
  (diffraction intensity).
- ``dataset_functions``: registry mapping dataset key (``"aunp"``, ``"chart"``,
  ``"simu"``, ``"mgall"``) to its loader function. Loaders read the experimental
  diffraction stack from disk based on the config's ``sample`` key.
- ``load_*_data``: individual dataset loaders, exported for direct use.
"""

from pid3net.utils.datagenerator_ssp import DataIteratorSsp
from pid3net.utils.general import (
    dataset_functions,
    load_aunp_data,
    load_chart_data,
    load_mg_data,
    load_simu_data,
)

__all__ = [
    "DataIteratorSsp",
    "dataset_functions",
    "load_aunp_data",
    "load_chart_data",
    "load_mg_data",
    "load_simu_data",
]
