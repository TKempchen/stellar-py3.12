import torch
import torch.nn as nn
import os
import os.path
import random
import numpy as np
import torch.nn.functional as F

def prepare_save_dir(args, filename):
    """ Create saving directory."""
    runner_name = os.path.basename(filename).split(".")[0]
    model_dir = './experiments/{}/{}/'.format(runner_name, args.name)
    args.savedir = model_dir
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    return args

def set_seed(seed):
    """Seed every RNG STELLAR draws from. Must run before any model construction."""
    random.seed(seed)
    np.random.seed(seed)          # STELLAR.train_epoch -> np.random.choice
    torch.manual_seed(seed)       # weight init + ClusterLoader(shuffle=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_device(choice):
    """Resolve the --device argument to a torch.device.

    'auto' resolves to cuda if available, else cpu -- deliberately NOT mps.
    MPS float32 math diverges from CPU/CUDA and torch.manual_seed drives a
    separate RNG stream there, so silently defaulting to it would violate
    algorithmic equivalence with the original code path. Use --device mps
    as an explicit, documented opt-in.
    """
    if choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif choice == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested but CUDA is not available')
        return torch.device('cuda')
    elif choice == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('--device mps requested but MPS is not available')
        return torch.device('mps')
    elif choice == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError('Unknown --device choice: %r' % (choice,))

def resolve_louvain_flavor(choice='auto'):
    """Resolve the scanpy louvain flavor STELLAR.train() should use.

    'igraph' uses scanpy's igraph branch (igraph's own community_multilevel).
    It needs nothing beyond python-igraph, which scanpy already depends on, so
    it is the only flavor that installs cleanly on every platform. Note that
    scanpy ignores `resolution` for this flavor.
    'vtraag' is scanpy's default and needs the `louvain` package, which pins
    igraph <0.12 (so it cannot coexist with leidenalg), has no macOS arm64
    wheel, and imports `pkg_resources` at runtime -- which setuptools >=81 no
    longer ships.
    'taynaud' uses python-louvain, which is pure Python with no igraph
    dependency at all, so it installs anywhere.

    The three are NOT interchangeable in general: they run Louvain on different
    graph representations (taynaud undirected+weighted, vtraag directed+
    unweighted) and land on different partitions. That only affects STELLAR's
    output when num_seed_class > 0: at num_seed_class == 0, est_seeds discards
    the clustering entirely (it returns all zeros regardless of `clusters`), so
    the choice is immaterial there.

    'auto' prefers vtraag and falls back to igraph with a warning, so the
    substitution is never silent.
    """
    if choice == 'auto':
        try:
            import louvain  # noqa: F401
            return 'vtraag'
        except Exception:
            # Deliberately broad: a broken `louvain` install fails in ways other
            # than ImportError (e.g. ModuleNotFoundError for pkg_resources under
            # setuptools >=81), and 'auto' must degrade rather than crash.
            print("WARNING: the 'louvain' package is unavailable or unusable; "
                  "falling back to flavor='igraph' (igraph community_multilevel). "
                  "Partitions will differ from vtraag, which changes results when "
                  "--num-seed-class > 0. At --num-seed-class 0 the clustering is "
                  "discarded, so results are unaffected.")
            return 'igraph'
    if choice == 'igraph':
        try:
            import igraph  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "--louvain-flavor igraph requires python-igraph "
                "(pip install igraph)."
            ) from e
        return 'igraph'
    if choice == 'vtraag':
        try:
            import louvain  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "--louvain-flavor vtraag requires a working 'louvain' package. It "
                "pins igraph <0.12 (conflicting with leidenalg), has no macOS "
                "arm64 wheel, and needs setuptools <81 for pkg_resources. Use "
                "'igraph' instead, or a dedicated environment."
            ) from e
        return 'vtraag'
    if choice == 'taynaud':
        try:
            import community  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "--louvain-flavor taynaud requires python-louvain >=0.16 "
                "(pip install 'python-louvain>=0.16')."
            ) from e
        return 'taynaud'
    raise ValueError('Unknown --louvain-flavor choice: %r' % (choice,))

def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)