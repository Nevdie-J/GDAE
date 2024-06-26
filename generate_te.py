import argparse

import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import ClusterData
from utils import *
from datasets import *
from torch_sparse import SparseTensor

def generate_topology_encoding(data, encoding_size):
    A = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.ones(data.edge_index.size(1)), sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    D_inv = (A.sum(1).squeeze() + 1e-10) ** -1.0
    
    I = torch.eye(data.x.shape[0], device=data.x.device)
    row, col = dense_to_sparse(I)[0]
    D_inv = SparseTensor(row=row, col=col, value=D_inv, sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    
    P = A @ D_inv
    M = P
    
    TE = [M.get_diag().float()]
    M_power = M
    for _ in tqdm(range(encoding_size - 1)):
        M_power = M_power @ M
        TE.append(M_power.get_diag().float())
    TE = torch.stack(TE, dim=-1)

    return TE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dataset', type=str, default='cora')
    parser.add_argument('-encoding_size', type=int, default=32)
    
    args = parser.parse_args()
    
    data, articulation_points, bridges, components, y_ap = load_data(name=args.dataset)
    pe = generate_topology_encoding(data, args.encoding_size)

        
    TE = pe
    print(TE.shape)
    torch.save(TE, f'./data/topology_encodings/{args.dataset}_te.pt')
    