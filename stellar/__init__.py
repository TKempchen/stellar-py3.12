"""STELLAR: annotation of spatially resolved single-cell data.

Brbić, M., Cao, K., Hickey, J.W. et al. Annotation of spatially resolved
single-cell data with STELLAR. Nat Methods 19, 1411-1418 (2022).
https://doi.org/10.1038/s41592-022-01651-8

The modules used to sit at the repository root as flat scripts (``STELLAR.py``,
``datasets.py``, ``utils.py``, ``models/``), imported by appending the checkout
to ``sys.path``. They are nested under this package so the project is
pip-installable and does not claim the generic top-level names ``utils``,
``datasets`` and ``models``. The algorithm itself is unchanged.
"""

from .STELLAR import STELLAR
from .datasets import (
    GraphDataset,
    get_hubmap_edge_index,
    get_tonsilbe_edge_index,
    load_hubmap_data,
    load_tonsilbe_data,
)
from .utils import (
    MarginLoss,
    entropy,
    prepare_save_dir,
    resolve_louvain_flavor,
    select_device,
    set_seed,
)

__all__ = [
    "STELLAR",
    "GraphDataset",
    "get_hubmap_edge_index",
    "get_tonsilbe_edge_index",
    "load_hubmap_data",
    "load_tonsilbe_data",
    "MarginLoss",
    "entropy",
    "prepare_save_dir",
    "resolve_louvain_flavor",
    "select_device",
    "set_seed",
]

__version__ = "0.2.0"
