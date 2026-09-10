# Changelog

All notable changes between the upstream `main` branch
([snap-stanford/stellar](https://github.com/snap-stanford/stellar), commit `13c81ad`)
and this working tree (branch `updates`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-10

### Summary

Port of STELLAR from Python 3.8 / PyTorch 1.9.1 / PyTorch Geometric 2.0 to
Python 3.12 / PyTorch 2.5.1 / PyTorch Geometric 2.8.0, and restructuring of the
repository from flat root-level scripts into a pip-installable `stellar` package.
The training and inference algorithm is unchanged; the code changes are import
paths, three new CLI knobs (`--louvain-flavor`, `--num-parts`, `--device`), and
RNG seeding that now actually takes effect.

File-level overview (`git diff main --stat` plus untracked files):

| Path                                                   | Status                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `.gitignore`                                           | modified                                                            |
| `README.md`                                            | modified                                                            |
| `STELLAR_run.py`                                       | modified                                                            |
| `requirements.txt`                                     | modified (fully rewritten)                                          |
| `STELLAR.py`                                           | deleted at root, moved to `stellar/STELLAR.py` **with edits**       |
| `datasets.py`                                          | deleted at root, moved to `stellar/datasets.py` **with one edit**   |
| `utils.py`                                             | deleted at root, moved to `stellar/utils.py` **with additions**     |
| `models/__init__.py`, `models/net.py`                  | copied byte-identical to `stellar/models/`; root copy still present |
| `stellar/__init__.py`                                  | new                                                                 |
| `pyproject.toml`                                       | new                                                                 |
| `LICENSE`, `demo.ipynb`, `images/stellar_overview.png` | unchanged                                                           |

---

### Added

#### `stellar/` package

- **`stellar/__init__.py`** (new). Module docstring cites the paper
  (Brbić et al., *Nat Methods* 19, 1411–1418, 2022, doi:10.1038/s41592-022-01651-8)
  and explains that the former root-level scripts were nested under a package so the
  project is pip-installable and no longer claims the generic top-level names
  `utils`, `datasets` and `models`.
  Re-exports (and lists in `__all__`):
  
  - from `.STELLAR`: `STELLAR`
  - from `.datasets`: `GraphDataset`, `get_hubmap_edge_index`, `get_tonsilbe_edge_index`,
    `load_hubmap_data`, `load_tonsilbe_data`
  - from `.utils`: `MarginLoss`, `entropy`, `prepare_save_dir`, `resolve_louvain_flavor`,
    `select_device`, `set_seed`
  - Sets `__version__ = "0.2.0"`.

- **`stellar/models/__init__.py`, `stellar/models/net.py`** — byte-identical copies of
  the root `models/` package (verified by SHA-1: `7333724…` and `d2bd3e7…` respectively
  match `main`).

#### New functions in `stellar/utils.py`

All three are pure additions; no existing line of `utils.py` was removed or modified
(103 lines added, 0 removed). Two new imports support them: `import random`,
`import numpy as np`.

- **`set_seed(seed)`** — seeds every RNG STELLAR draws from:
  `random.seed`, `np.random.seed` (used by `STELLAR.train_epoch` via `np.random.choice`),
  `torch.manual_seed` (weight init and `ClusterLoader(shuffle=True)`), and
  `torch.cuda.manual_seed_all` when CUDA is available. Docstring notes it must run
  before any model construction.

- **`select_device(choice)`** — resolves the `--device` CLI value to a `torch.device`:
  
  - `'auto'` → `cuda` if available, else `cpu`. Deliberately never resolves to `mps`
    (MPS float32 math and its RNG stream diverge from CPU/CUDA; docstring explains this
    would break algorithmic equivalence with the original code path).
  - `'cuda'` → raises `RuntimeError('--device cuda requested but CUDA is not available')` if unavailable.
  - `'mps'` → raises `RuntimeError('--device mps requested but MPS is not available')` if unavailable.
  - `'cpu'` → `torch.device('cpu')`.
  - anything else → `ValueError`.

- **`resolve_louvain_flavor(choice='auto')`** — resolves the scanpy `flavor` argument used
  by the seed-clustering step in `STELLAR.train()`:
  
  - `'auto'`: tries `import louvain`; on success returns `'vtraag'` (scanpy's default and
    the flavor upstream STELLAR effectively used). On **any** exception (deliberately
    broad, because a broken `louvain` can fail with e.g. `ModuleNotFoundError` for
    `pkg_resources` under setuptools ≥81) prints a `WARNING:` line and returns `'igraph'`.
  - `'igraph'`: verifies `import igraph`, else `RuntimeError` with `pip install igraph` hint.
  - `'vtraag'`: verifies `import louvain`, else `RuntimeError` explaining the igraph <0.12
    pin, missing macOS arm64 wheel, and setuptools <81 requirement.
  - `'taynaud'`: verifies `import community`, else `RuntimeError` pointing at
    `python-louvain>=0.16`.
  - anything else → `ValueError`.
  - Docstring documents that the flavors run Louvain on different graph representations
    and yield different partitions, but that this only affects output when
    `num_seed_class > 0` (at 0, `est_seeds` discards the clustering and returns all zeros).

#### New CLI arguments in `STELLAR_run.py`

- `--louvain-flavor {auto,igraph,vtraag,taynaud}` (default `auto`) — passed through to
  `resolve_louvain_flavor`.
- `--num-parts INT` (default `100`) — number of METIS partitions for `ClusterData`;
  help text warns that too few nodes per partition can yield empty-edge partitions that
  crash `ClusterLoader`.
- `--device {auto,cuda,mps,cpu}` (default `auto`) — passed through to `select_device`;
  help text states `auto` never picks `mps`.

#### Packaging and environment files

- **`pyproject.toml`** (new):
  
  - `[build-system]`: `setuptools>=64`, `wheel`, backend `setuptools.build_meta`.
  - `[project]`: name `stellar-gnn`, version `0.2.0`, description
    "STELLAR: annotation of spatially resolved single-cell data (Python 3.12 port)",
    `readme = "README.md"`, `license = { file = "LICENSE" }`, `requires-python >= 3.10`,
    authors Kaidi Cao / Maria Brbic / John W. Hickey, maintainer Tim Kempchen,
    keywords, classifiers for Python 3.10–3.12, MIT, Science/Research, Bio-Informatics.
  - `dependencies`: `torch`, `torch-geometric>=2.4`, `numpy`, `pandas`, `scipy`,
    `scikit-learn`, `anndata`, `scanpy` (all unpinned).
  - Deliberately **excluded** (with an explanatory comment): `torch-sparse` and
    `torch-scatter` (required at runtime, but PyPI only has generic sdists that would
    build against the wrong ABI; must come from `data.pyg.org`), and
    `louvain` / `python-louvain` (the default flavor only needs `python-igraph`, already a
    scanpy dependency).
  - `[project.urls]`: Homepage `https://github.com/TKempchen/STELLAR-dev`,
    Upstream `https://github.com/snap-stanford/stellar`, Paper DOI.
  - `[tool.setuptools.packages.find]`: `include = ["stellar*"]`, `namespaces = false`.

- **`.gitignore`** — added `*.egg-info/`, `build/`, `dist/`, `__pycache__/`, `*.pyc`
  (previously only `.DS_Store`).

- **`README.md`** — fork note and per-platform install instructions (details under *Changed*).

---

### Changed

#### `STELLAR.py` → `stellar/STELLAR.py` (moved **and** edited)

Not a pure move. 19 substantive line changes plus one whitespace-only cleanup
(trailing spaces on a blank line inside `train_supervised` removed). Algorithm,
loss functions, `est_seeds`, `pred`, and all hyperparameters are unchanged.

1. Imports made package-relative and extended:
   
   ```diff
   -import models
   -from utils import entropy, MarginLoss
   +from . import models
   +from .utils import entropy, MarginLoss, resolve_louvain_flavor
   ```
2. PyTorch Geometric import path updated to the post-2.0 location:
   
   ```diff
   -from torch_geometric.data import ClusterData, ClusterLoader
   +from torch_geometric.loader import ClusterData, ClusterLoader
   ```
3. `train_supervised`: `num_parts` read from `args` with the original value as fallback:
   
   ```diff
   +        num_parts = getattr(self.args, 'num_parts', 100)
            labeled_graph = dataset.labeled_data
   -        labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
   +        labeled_data = ClusterData(labeled_graph, num_parts=num_parts, recursive=False)
   ```
4. `train_epoch`: same change applied to both `ClusterData` calls (labeled and unlabeled):
   
   ```diff
   +        num_parts = getattr(self.args, 'num_parts', 100)
            labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
   -        labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
   +        labeled_data = ClusterData(labeled_graph, num_parts=num_parts, recursive=False)
            ...
   -        unlabeled_data = ClusterData(unlabeled_graph, num_parts=100, recursive=False)
   +        unlabeled_data = ClusterData(unlabeled_graph, num_parts=num_parts, recursive=False)
   ```
5. `train()`: Louvain flavor is now resolved explicitly instead of relying on scanpy's
   default (`vtraag`):
   
   ```diff
   -        sc.tl.louvain(adata, 1)
   +        # resolution=1 is passed positionally as before. Note scanpy ignores it for
   +        # flavor='taynaud', but python-louvain's best_partition defaults to 1.0 anyway.
   +        flavor = resolve_louvain_flavor(getattr(self.args, 'louvain_flavor', 'auto'))
   +        sc.tl.louvain(adata, 1, flavor=flavor)
   ```
   
   Because `getattr(..., 'auto')` and `getattr(..., 100)` are used, the class still works
   with an `args` namespace that lacks the new attributes (e.g. from `demo.ipynb` or
   external callers such as SPACEc).

#### `datasets.py` → `stellar/datasets.py` (moved **and** edited)

Exactly one line removed, nothing else changed:

```diff
-from builtins import range
```

(Python 2 compatibility shim from the `future` package; a no-op on Python 3.)

#### `utils.py` → `stellar/utils.py` (moved **and** extended)

Additions only — see *Added*. `prepare_save_dir`, `entropy`, and `MarginLoss` are
byte-for-byte unchanged.

#### `models/` → `stellar/models/` (pure copy)

Both files identical to `main`. The root `models/` directory has **not** been deleted
(see *Known issues*).

#### `STELLAR_run.py`

1. Imports rewired to the package:
   
   ```diff
   -from utils import prepare_save_dir
   -from STELLAR import STELLAR
   +from stellar.utils import prepare_save_dir, set_seed, select_device
   +from stellar import STELLAR
    ...
   -from datasets import GraphDataset, load_tonsilbe_data, load_hubmap_data
   +from stellar.datasets import GraphDataset, load_tonsilbe_data, load_hubmap_data
   ```
2. Three new `argparse` arguments added after `--savedir` (listed under *Added*).
3. Device / seeding block replaced:
   
   ```diff
   -    args.cuda = torch.cuda.is_available()
   -    args.device = torch.device("cuda" if args.cuda else "cpu")
   -
   -    # Seed the run and create saving directory
   +    # Seed every RNG before any model construction (weight init happens in STELLAR.__init__).
   +    set_seed(args.seed)
   +
   +    args.device = select_device(args.device)
   +    args.cuda = args.device.type == 'cuda'
   +
   +    # Create saving directory
   ```
   
   `args.cuda` is kept for backward compatibility with any code that reads it.
4. Dataset loading, `STELLAR(args, dataset)`, `train()`, `pred()`, and the
   `np.save(...)` of results are unchanged.

#### `requirements.txt` (fully rewritten)

Every original pin was replaced, two packages were dropped, and four PyTorch-stack
packages were added. One file serves both platforms: torch is installed first for the
platform, then `pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+<cpu|cu124>.html`
supplies the ABI-matched torch-scatter / torch-sparse wheels. `louvain` is present
but commented out (optional source build; STELLAR falls back to the `igraph` flavor).

| Package         | `main` | now                    | Note                                                                                               |
| --------------- | ------ | ---------------------- | -------------------------------------------------------------------------------------------------- |
| torch           | —      | 2.5.1                  | added                                                                                              |
| torch-geometric | —      | 2.8.0                  | added                                                                                              |
| torch-sparse    | —      | 0.6.18                 | added; required for `ClusterData`'s METIS backend                                                  |
| torch-scatter   | —      | 2.1.2                  | added; `torch_sparse/storage.py` imports it unconditionally                                        |
| numpy           | —      | 1.26.4                 | added; pinned `<2` to avoid NEP-50 promotion changes in `pairwise_distances` near `distance_thres` |
| pandas          | 1.3    | 2.2.3                  |                                                                                                    |
| scipy           | 1.7    | 1.14.1                 |                                                                                                    |
| scikit-learn    | 1.0.2  | 1.5.2                  |                                                                                                    |
| anndata         | 0.7.6  | 0.11.4                 |                                                                                                    |
| scanpy          | 1.8    | 1.11.1                 |                                                                                                    |
| igraph          | 0.9.10 | 0.11.8                 |                                                                                                    |
| louvain         | 0.7.1  | (0.8.2, commented out) | optional; source build only (no cp312 wheel)                                                       |
| python-louvain  | 0.1    | —                      | **dropped**: never imported; only reachable via scanpy `flavor='taynaud'`                          |
| scikit-image    | 0.18   | —                      | **dropped**: never imported anywhere in the repo                                                   |

#### `README.md`

- New one-line fork note under the title (fork of snap-stanford/stellar, targets Python 3.12).
- Installation keeps upstream's four-step layout; content updated:
  1. `python=3.8` → `python=3.12`; `source activate` → `conda activate`.
  2. **Pytorch**: `pip install torch==2.5.1` (macOS) or
     `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124` (Linux, CUDA 12.4)
     replaces `conda install pytorch cudatoolkit=11.3 -c pytorch`.
  3. **Pytorch Geometric and other dependencies**:
     `pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+{cpu,cu124}.html`
     replaces `conda install pyg -c pyg` + `pip install -r requirements.txt`.
  4. **STELLAR**: `pip install .`, plus a one-liner verifying `torch_geometric.typing.WITH_TORCH_SPARSE`.
- Tested-platforms note: "NVIDIA GPU, Linux, Python3 … Ubuntu 16.04 with NVIDIA Geforce
  2080 Ti GPU and 1T CPU memory … macOS (Intel chip)" → "Python 3.12, PyTorch 2.5.1 and
  PyTorch Geometric 2.8.0 on Linux with an NVIDIA GPU (CUDA 12.4) and on macOS
  (Apple Silicon, CPU)"; adds that `louvain` is optional (`--louvain-flavor igraph` fallback).
- "Getting started" code block: fence now tagged `python` and prefixed with
  `from stellar import STELLAR, GraphDataset`; the three usage lines are unchanged.
- Title, overview, figure, and everything before the Installation section unchanged.

---

### Removed

- Root-level `STELLAR.py`, `datasets.py`, `utils.py` (relocated into `stellar/`).
- `from builtins import range` in `datasets.py`.
- `python-louvain` and `scikit-image` from `requirements.txt`.
- README installation steps for PyTorch 1.9.1 / cudatoolkit 11.3 and `conda install pyg`.

---

### Behavior changes (user-visible)

1. **Seeding now works.** On `main`, `--seed` was parsed but never used; no RNG was ever
   seeded, so runs were non-reproducible. `set_seed(args.seed)` is now called before
   `STELLAR.__init__` constructs the model. Default seed (`1`) is unchanged.
2. **Louvain flavor is explicit.** Upstream implicitly used scanpy's default `vtraag`
   (the `louvain` package). With `--louvain-flavor auto` (default), STELLAR still uses
   `vtraag` when `louvain` imports cleanly, but otherwise prints a warning and uses
   `igraph`. Partitions differ between flavors; results change **only** when
   `--num-seed-class > 0`.
3. **Device selection is explicit.** `--device auto` behaves exactly like the old code
   (cuda if available, else cpu). `mps` is available only by explicit opt-in.
   Requesting an unavailable device now raises instead of silently falling back.
4. **`num_parts` is configurable.** Default remains 100, so default behavior is identical.
5. **Import paths.** `from STELLAR import STELLAR` / `from utils import …` /
   `from datasets import …` no longer work; use `from stellar import …`.
6. **Installable.** `pip install .` now works via `pyproject.toml` (distribution name
   `stellar-gnn`, import name `stellar`).
