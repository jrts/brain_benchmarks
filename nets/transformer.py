import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers.gcn_layer import GCNLayer
from pooling.sagpool import SAGPoolBlock
from pooling.hgpslpool import HGPSLPoolBlock

class GraphTransformer(nn.Module):

    def __init__(self, net_params):
        super(GraphTransformer, self).__init__()
        in_dim = net_params['in_dim']
        edge_dim = net_params['edge_dim']
        hidden_dim = net_params['hidden_dim']
        dropout = net_params['dropout']
        in_feat_dropout = net_params['in_feat_dropout']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.e_feat = net_params['edge_feat']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.feature_dim_size = hidden_dim
        self.ff_hidden_size = 256
        self.num_classes = net_params['n_classes']
        self.num_self_att_layers = 3  # Each layer consists of a number of self-attention layers
        self.num_GNN_layers = 2
        self.nhead = 1
        self.lst_gnn = torch.nn.ModuleList()
        #
        self.ugformer_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=self.nhead, dim_feedforward=self.ff_hidden_size, dropout=0.5)  # Default batch_first=False (seq, batch, feature)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            self.lst_gnn.append(GCNLayer(self.feature_dim_size, self.feature_dim_size, F.relu, dropout, self.batch_norm, self.residual, e_feat=self.e_feat))

        # Linear function
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

        self.prediction = nn.Linear(self.feature_dim_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, e):

        h = self.embedding_h(h)
        # e = self.embedding_e(e)
        h = self.in_feat_dropout(h)

        prediction_scores = torch.zeros([g.batch_size, self.num_classes], device=h.device)
        for layer_idx in range(self.num_GNN_layers):
            input_Tr = h
            # self-attention over all nodes
            # print(input_Tr.shape)
            input_Tr = torch.unsqueeze(input_Tr, 1)  #[seq_length, batch_size=1, dim] for pytorch transformer
            input_Tr = self.ugformer_layers[layer_idx](input_Tr)
            # print(input_Tr.shape)
            input_Tr = torch.squeeze(input_Tr, 1)
            # take a sum over neighbors followed by a linear transformation and an activation function --> similar to GCN
            h, e = self.lst_gnn[layer_idx](g, input_Tr, e)
            # # graph_embedding = torch.sum(input_Tr, dim=0)
            # graph_embedding = self.dropouts[layer_idx](hg)
            # # Produce the final scores
            # prediction_scores += self.predictions[layer_idx](graph_embedding)
        # Can modify the code by commenting Lines 48-51 and uncommenting Lines 33-34, 53-56 to only use the last layer to make a prediction.
        g.ndata['h'] = h
        # take a sum over all node representations to get graph representations
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        # graph_embedding = torch.sum(input_Tr, dim=0)
        graph_embedding = self.dropout(hg)
        # Produce the final scores
        prediction_scores = self.prediction(graph_embedding)

        return prediction_scores

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss