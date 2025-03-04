# import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
import dgl
# from pooling.sagpool import SAGPoolBlock
# from pooling.hgpslpool import HGPSLPoolBlock

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        edge_dim = net_params['edge_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']    
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.e_feat = net_params['edge_feat']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu, dropout, aggregator_type,
                                                    batch_norm, residual, e_feat=self.e_feat) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm,
                                          residual, e_feat=self.e_feat))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        e = self.embedding_e(e)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        scores = self.MLP_layer(hg)
        return scores


    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss
