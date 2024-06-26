import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx

from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import to_networkx, remove_self_loops, train_test_split_edges, to_dense_adj, add_self_loops, negative_sampling, dropout_adj, add_remaining_self_loops, degree, to_undirected
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_sparse import SparseTensor

warnings.simplefilter("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)

def get_split(num_samples, train_ratio = 0.1, test_ratio = 0.8, num_splits = 10):
    
    # random split
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    mask.unsqueeze(dim=-1)
    return mask

def load_syn_data(type='attribute'):

    edge_index = torch.load(f'./data/synthetic/{type}/edge_index.pt')
    edge_index, _ = add_remaining_self_loops(edge_index)
    x = torch.load(f'./data/synthetic/{type}/x.pt')
    y = torch.load(f'./data/synthetic/{type}/y.pt')

    num_label = len(torch.unique(y))

    # 训练测试划分
    num_per_label = 20
    train_index, val_index, test_index = [], [], []
    for i in range(num_label):
        index = (y.long() == i).nonzero().view(-1)
        index = index[torch.randperm(index.shape[0])]
        train_index.append(index[:num_per_label])
        val_index.append(index[num_per_label:num_per_label*3])
        test_index.append(index[num_per_label*3:])
    train_index = torch.cat(train_index)
    val_index = torch.cat(val_index)
    test_index = torch.cat(test_index)

    train_mask = index_to_mask(train_index, x.shape[0])
    val_mask = index_to_mask(val_index, x.shape[0])
    test_mask = index_to_mask(test_index, x.shape[0])

    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return [data]

def load_data(root='/workspace/data', name='cora', is_random_split=False, only_data=False):
    if name in ['cora', 'citeseer']:
        dataset = Planetoid(root=root + '/Planetoid', name=name)
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=root + '/WebKB', name = name)
    elif name in ['actor']:
        dataset = Actor(root=root + '/Actor')
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root + '/WikipediaNetwork', name=name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=root + '/Amazon', name=name, transform=T.NormalizeFeatures())
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root=root + '/Coauthor', name=name, transform=T.NormalizeFeatures())
    elif name in ['wiki-cs']:
        dataset = WikiCS(root=root + '/Wiki-CS')
    elif name in ['attribute', 'topology', 'tl-80', 'tl-60', 'tl-40']:
        dataset = load_syn_data(type=name)
    data = dataset[0]
    
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = to_undirected(data.edge_index)
    data.edge_sparse = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.ones(data.edge_index.size(1)), sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    
    # 数据集划分
    if name in ['computers', 'photo', 'cs', 'physics', 'wiki-cs'] or is_random_split == True:
        train_mask, val_mask, test_mask = get_split(data.x.shape[0])
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
    if len(val_mask.shape) < 2:
        val_mask = val_mask.unsqueeze(1)
    if len(test_mask.shape) < 2:
        test_mask = test_mask.unsqueeze(1)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    data.degrees = degree(data.edge_index[0], data.x.shape[0])
    data.degree_centrality = data.degrees / (data.x.shape[0] - 1)
    
    articulation_points, bridges, components, y_ap = None, None, None, None
    if only_data==False:
        # biconnectivity
        G = to_networkx(data).to_undirected()
        articulation_points = list(nx.articulation_points(G))
        bridges = list(nx.bridges(G))
        components = list(nx.connected_components(G))
        ap_idx = torch.tensor(articulation_points)
        y_ap = torch.zeros(len(data.y))
        if len(ap_idx):
            y_ap[ap_idx] = 1
            
    return data, articulation_points, bridges, components, y_ap
