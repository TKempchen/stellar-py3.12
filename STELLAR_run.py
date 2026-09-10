import argparse
from stellar.utils import prepare_save_dir, set_seed, select_device
from stellar import STELLAR
import numpy as np
import os
import torch
from stellar.datasets import GraphDataset, load_tonsilbe_data, load_hubmap_data

def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--dataset', default='Hubmap', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='STELLAR')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--num-heads', type=int, default=22)
    parser.add_argument('--num-seed-class', type=int, default=0)
    parser.add_argument('--sample-rate', type=float, default=0.5)
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size')
    parser.add_argument('--distance_thres', default=50, type=int)
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--louvain-flavor', type=str, default='auto',
                    choices=['auto', 'igraph', 'vtraag', 'taynaud'],
                    help="scanpy louvain flavor. 'igraph' needs only python-igraph and "
                         "works on every platform. 'auto' prefers vtraag (the `louvain` "
                         "package) and falls back to igraph with a warning. They give "
                         "different partitions, which only changes results when "
                         "--num-seed-class > 0.")
    parser.add_argument('--num-parts', type=int, default=100,
                    help='METIS partitions for ClusterData. Lower this for small/sparse '
                         'graphs -- too few nodes per partition relative to num_parts can '
                         'produce empty-edge partitions that crash ClusterLoader.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                    help="Compute device. 'auto' picks cuda if available, else cpu "
                         "(never mps automatically -- pass --device mps explicitly; "
                         "MPS float32 math and RNG stream diverge from CPU/CUDA).")
    args = parser.parse_args()

    # Seed every RNG before any model construction (weight init happens in STELLAR.__init__).
    set_seed(args.seed)

    args.device = select_device(args.device)
    args.cuda = args.device.type == 'cuda'

    # Create saving directory
    args.name = '_'.join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)
    
    if args.dataset == 'Hubmap':
        labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, inverse_dict = load_hubmap_data('./data/B004_training_dryad.csv', './data/B0056_unnanotated_dryad.csv', args.distance_thres, args.sample_rate)
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    elif args.dataset == 'TonsilBE':
        labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, inverse_dict = load_tonsilbe_data('./data/BE_Tonsil_l3_dryad.csv', args.distance_thres, args.sample_rate)
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges)
    stellar = STELLAR(args, dataset)
    stellar.train()
    _, results = stellar.pred()
    np.save(os.path.join(args.savedir, args.dataset + '_results.npy'), results)

if __name__ == '__main__':
    main()
