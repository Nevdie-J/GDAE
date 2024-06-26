import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv

def get_gnn_layer(name, in_channels, out_channels, heads):
    if name == "sage":
        layer = SAGEConv(in_channels, out_channels)
    elif name == "gcn":
        layer = GCNConv(in_channels, out_channels)
    elif name == "gin":
        layer = GINConv(Linear(in_channels, out_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, out_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, out_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer

def get_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def topology_recon_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss