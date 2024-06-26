import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_undirected, sort_edge_index, degree, negative_sampling, add_self_loops, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import scipy.sparse as sp
import os
from sklearn.neighbors import kneighbors_graph

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
    
from typing import Optional, Tuple

import torch.nn.functional as F
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv

from torch_sparse import SparseTensor

from sklearn.metrics import roc_auc_score, average_precision_score

from models.utils import get_gnn_layer, get_activation_layer, topology_recon_loss, to_sparse_tensor

class MLPEncoder(nn.Module):
    # individual learning from node-wise view
    def __init__(self, in_channels, pe_dim, hidden_channels, out_channels, dropout=0.5, norm=False, activation="tanh"):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)
        
        bn = nn.BatchNorm1d if norm else nn.Identity
        self.mlpX = nn.ModuleList()
        self.mlpP = nn.ModuleList()
        
        self.mlpX.append(nn.Linear(in_channels, hidden_channels))
        self.mlpX.append(nn.Linear(hidden_channels*2, hidden_channels))
        self.bnX1 = bn(hidden_channels)
        self.mlpX.append(nn.Linear(hidden_channels, out_channels))
        self.bnX2 = bn(out_channels)
        
        self.mlpP.append(nn.Linear(pe_dim, hidden_channels))
        self.mlpP.append(nn.Linear(hidden_channels, out_channels))
        self.bnP = bn(out_channels)

    def forward(self, x, p):
        
        x = self.activation(self.mlpX[0](x))
        p = self.activation(self.mlpP[0](p))
        
        x = torch.cat([x, p], dim=-1)
        x = self.mlpX[1](x)
        x = self.activation(self.bnX1(x))
        
        x = self.dropout(x)
        p = self.dropout(p)
        
        x = self.mlpX[2](x)
        x = self.activation(self.bnX2(x))
        
        p = self.mlpP[1](p)
        p = self.activation(self.bnP(p))
        
        return x, p

    @torch.no_grad()
    def get_node_embedding(self, x, p, mode='last'):
        self.eval()
        
        out = []
        x = self.activation(self.mlpX[0](x))
        out.append(x)
        p = self.activation(self.mlpP[0](p))
        
        x = torch.cat([x, p], dim=-1)
        x = self.mlpX[1](x)
        x = self.activation(self.bnX1(x))
        out.append(x)
        
        x = self.dropout(x)
        p = self.dropout(p)
        
        x = self.mlpX[2](x)
        x = self.activation(self.bnX2(x))
        out.append(x)
        
        p = self.mlpP[1](p)
        p = self.activation(self.bnP(p))
        
        if mode == 'cat':
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]
        
        return embedding, p

class MPGNNEncoder(nn.Module):
    # collective learning from aggregated view
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, norm=False, layer="gcn", activation="elu"):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if norm else nn.Identity
        
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(get_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels*heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x, edge_index):
        edge_sparse = to_sparse_tensor(edge_index, x.size(0))
        
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_sparse)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_sparse)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x
    
    @torch.no_grad()
    def get_node_embedding(self, x, edge_index, mode="last"):

        self.eval()

        edge_sparse = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_sparse)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_sparse)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding

class NodeDecoder(nn.Module):
    # attribute decoding
    def __init__(self, in_channels, hidden_channels, out_channels, activation='elu'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )
        if activation == None:
            self.activation = activation
        else:
            self.activation = get_activation_layer(activation)
    
    def forward(self, x):
        decoding = self.mlp(x)
        if self.activation != None:
            decoding = self.activation(decoding)
        return decoding

    def sce_loss(self, x, y, alpha=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

class EdgeDecoder(nn.Module):
    # topology decoding
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.5, activation='relu'):
        super().__init__()
        
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels1 = in_channels*2 if i == 0 else hidden_channels
            second_channels1 = out_channels if i == num_layers - 1 else hidden_channels
        
            self.mlps.append(nn.Linear(first_channels1, second_channels1))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, z2, edge, sigmoid=True):
        
        x = z
        x = x[edge[0]] * x[edge[1]]

        x2 = z2
        x2 = x2[edge[0]] * x2[edge[1]]

        x = torch.cat([x, x2], dim=-1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
            
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x
    
class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.3, activation='elu'):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.bn = nn.BatchNorm1d(hidden_channels)

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x):

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.bn(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        x = self.activation(x)

        return x
        

class Masker(nn.Module):
    def __init__(self, p=0.7, undirected=True, walks_per_node=1, walk_length=3, num_nodes=None):
        super().__init__()
        self.p = p
        self.undirected = undirected
        
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.num_nodes = num_nodes
    
    def mask_edges(self, edge_index, p, walks_per_node=1, walk_length=3, num_nodes=None, training=True):
        
        random_walk = torch.ops.torch_cluster.random_walk
        
        edge_mask = edge_index.new_ones(edge_index.shape[1], dtype=torch.bool)
        
        if not training or p == 0.0:
            return edge_index, edge_mask

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = edge_index
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes*p)].repeat(walks_per_node)
        
        deg = degree(row, num_nodes=num_nodes)
        rowptr = row.new_zeros(num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])
        n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
        e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
        edge_mask[e_id] = False

        return edge_index[:, edge_mask], edge_index[:, ~edge_mask]

    def forward(self, edge_index):
        remaining_edges, masked_edges = self.mask_edges(edge_index, self.p, 
                                                self.walks_per_node,
                                                walk_length=self.walk_length,
                                                num_nodes=self.num_nodes
                                                )
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
            
        return remaining_edges, masked_edges


class GDAEModel(nn.Module):
    def __init__(self, mpgnn_encoder, edge_decoder, node_decoder, mlp_encoder, degree_decoder, mask=None):
        super().__init__()
        self.mpgnn_encoder = mpgnn_encoder
        self.edge_decoder = edge_decoder
        self.node_decoder = node_decoder
        self.mlp_encoder = mlp_encoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.top_loss = topology_recon_loss
    
    def reset_parameters(self):
        self.mpgnn_encoder.reset_parameters()
        self.mlp_encoder.reset_parameters()
        self.edge_decoder.reset_parameters()
        self.node_decoder.reset_parameters()
        self.degree_decoder.reset_parameters()
    
    def train_epoch(self, data, optimizer, batch_size=2 ** 16, grad_norm=1.0, lam1=0.1, lam2=0.001):
        
        x, edge_index = data.x, data.edge_index
        
        # masking A
        if self.mask is not None:
            remaining_edges, masked_edges = self.mask(edge_index)
        else:
            remaining_edges = edge_index
            masked_edges = getattr(data, "pos_edge_label_index", edge_index)
        
        loss_total = 0.0
        edge_index, _ = add_self_loops(edge_index)
        neg_edges = negative_sampling(
            edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        x0 = torch.cat([x, data.pe], dim=-1)
        x, pe = x, data.pe

        # training
        for perm in DataLoader(
            range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ):

            optimizer.zero_grad()
            # dual-view encoding
            z = self.mpgnn_encoder(x, remaining_edges)
            z2, p = self.mlp_encoder(x, pe)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            # 边解码
            pos_out1 = self.edge_decoder(
                z, p, batch_masked_edges, sigmoid=False
            )
            neg_out1 = self.edge_decoder(z, p, batch_neg_edges, sigmoid=False)
            loss = self.top_loss(pos_out1, neg_out1)
            
            # 点解码
            decoding = self.node_decoder(z2)
            loss1 = self.node_decoder.sce_loss(decoding, x0)

            # 度中心性解码
            h = self.degree_decoder(torch.cat([z, z2], dim=-1))
            loss2 = F.mse_loss(h, data.degree_centrality)

            loss = loss + lam1*loss1 + lam2*loss2

            loss.backward()
            
            if grad_norm > 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()

            loss_total += loss.item()

        return loss_total

    @torch.no_grad()
    def batch_eval_lp(self, z, p, edges, batch_size=2**16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds.append(self.edge_decoder(z, p, edge).squeeze().cpu())
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def eval_lp(self, z, p, pos_edge_index, neg_edge_index, batch_size=2**16):
        pos_pred = self.batch_eval_lp(z, p, pos_edge_index)
        neg_pred = self.batch_eval_lp(z, p, neg_edge_index)
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.shape[0])
        neg_y = neg_pred.new_zeros(neg_pred.shape[0])
        
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        
        return roc_auc_score(y, pred)*100, average_precision_score(y, pred)*100