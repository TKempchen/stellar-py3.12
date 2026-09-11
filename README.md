# Annotation of Spatially Resolved Single-cell Data with STELLAR

This is a fork of [snap-stanford/stellar](https://github.com/snap-stanford/stellar) updated to run on newer Python versions, targeting Python 3.12 (PyTorch 2.5, PyTorch Geometric 2.8). 

NOTE: Use with caution and validate against the original version in case of doubt! 

[Project website](http://snap.stanford.edu/stellar)

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. STELLAR takes as input annotated reference spatial single-cell dataset in which cells are assigned to their cell types, and unannotated spatial dataset in which cell types are unknown. STELLAR then generates annotations for the unannotated dataset. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.nature.com/articles/s41592-022-01651-8).

<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>

### Installation

**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n stellar python=3.12
conda activate stellar
```

**2. Pytorch:**
Install PyTorch 2.5.1 for your platform (the torch-scatter / torch-sparse wheels in step 3 are built against exactly this version).

```bash
# macOS (Apple Silicon, CPU)
pip install torch==2.5.1

# Linux with NVIDIA GPU, CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**3. Pytorch Geometric and other dependencies:**
Install the pinned dependencies in [requirements.txt](requirements.txt), passing the data.pyg.org index that matches your platform:

```bash
# macOS
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Linux with NVIDIA GPU, CUDA 12.4
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

For another CUDA version replace `cu124` in both URLs (see [pytorch.org](https://pytorch.org/get-started/locally/) and [data.pyg.org/whl](https://data.pyg.org/whl/)).

**4. STELLAR:**

```bash
pip install .
```

Verify:

```bash
python -c "import torch_geometric.typing as t; from torch_geometric.loader import ClusterData; print(t.WITH_TORCH_SPARSE)"
```

**Note:** We tested STELLAR with Python 3.12, PyTorch 2.5.1 and PyTorch Geometric 2.8.0 on Linux with an NVIDIA GPU (CUDA 12.4) and on macOS (Apple Silicon, CPU). The `louvain` package is optional; without it STELLAR uses scanpy's `igraph` flavor (`--louvain-flavor igraph`).

### Getting started

We implemented STELLAR model in a self-contained class. To make an instance and train STELLAR:

```python
from stellar import STELLAR, GraphDataset

stellar = STELLAR(args, dataset)
stellar.train()
_, results = stellar.pred()
```

### Datasets

CODEX multiplexed imaging datasets used in STELLAR are made available at [dryad](https://datadryad.org/stash/share/1OQtxew0Unh3iAdP-ELew-ctwuPTBz6Oy8uuyxqliZk). Our demo code assumes the data to be put under the folder `./data/ ` you create.

### Demo

We provide several training examples with this repo:

- To run STELLAR on the CODEX healthy intestine data

```bash
python STELLAR_run.py --dataset Hubmap --num-heads 22
```

- To run STELLAR on the CODEX tonsil and BE data:

```bash
python STELLAR_run.py --dataset TonsilBE --num-heads 13 --num-seed-class 3
```

Memory usage and time:

- STELLAR expects graph as input. There are many ways to construct a graph. In this code, we construct a graph based on a predefined threshold. This part takes 32G physical memory for the HuBMAP dataset and 256G for Tonsil/BE. The longest construction takes around 10 minutes.
- Given a graph, running STELLAR on GPU takes less then 5G memory for both datasets and can finish within a few minutes.

We also provided a jupyter notebook [demo.ipynb](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/demo.ipynb) that shows example of running STELLAR on a downsampled dataset. Please consider to downsample more if there is a memory issue, but note that the performance of the model would degrade as the training data gets less. For users with limited memory and potentially limited access to GPU, please set the ``use-processed-graph`` to True to load pre-processsed data and can finish with CPU in about 30 mins.

### Use your own dataset

STELLAR expects graph as input. In our code, we construct a graph based on a predefined threshold, but STELLAR can work with any meaninfully constructed graph. To use your own dataset, you just need to initialize [GraphDataset](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L77) and give it to the input to our [stellar function](https://github.com/snap-stanford/stellar/blob/main/STELLAR.py).

```
dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
stellar = STELLAR(args, dataset)
```

- labeled_X and unlabeled_X are node features (that is gene/protein expressions) matrices for the annotated reference dataset and target unannotated dataset, respectively. They should be numpy arrays with shape [num_nodes, num_node_features] (that is  [num_cells, num_genes]). 
- labeled_y defines annotations for the annotated reference dataset. It is a numpy array with shape [num_nodes,] (that is [num_cells]). Annotations are expected to be numerical class categories.
- labeled_edges and unlabeled_edges define the input graphs for the annotated reference dataset and target unannotated dataset, respectively. They are numpy array with a shape [2, num_edges] and they define edges of the graph. For each edge, the indices of the nodes that are connected with that edge should be given.

Example for HuBMAP dataset is shown in [load_hubmap_data](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L30) function, and for Tonsil/BE dataset in [load_tonsilbe_data](https://github.com/snap-stanford/stellar/blob/a556b5ef4fe43c512ccf092c1d06d73034dc8d4d/datasets.py#L53). These examples demonstrate how to  initialize these variables from a csv file. 

### Citing

If you find our code and research useful, please consider citing:

```
@article{stellar2022,
  title={Annotation of spatially resolved single-cell data with STELLAR},
  author={Brbi{\'c}, Maria and Cao, Kaidi and Hickey, John W and Tan, Yuqi and Snyder, Michael P and Nolan, Garry P and Leskovec, Jure},
  journal={Nature Methods},
  volume={19},
  number={11},
  pages={1411--1418},
  year={2022},
  publisher={Nature Publishing Group}
}
```
