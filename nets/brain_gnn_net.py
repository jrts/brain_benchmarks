import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index, remove_self_loops)
from torch_sparse import spspmm
from layers.brain_gnn_layer import MyNNConv


##########################################################################################################################
class BrainGNNNet(torch.nn.Module):
    def __init__(self, net_params):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(BrainGNNNet, self).__init__()

        self.indim = net_params['in_dim']
        nclass = net_params['n_classes']
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.dim4 = 256
        self.dim5 = 32
        self.ratio = 0.5
        self.k = 32
        self.R = net_params['in_dim']
        self.edge_dim = net_params['edge_dim']

        if self.edge_dim > 1:
            self.edge_lin = nn.Linear(self.edge_dim, 1)
            self.edge_act = nn.Sigmoid()

        self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=self.ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=self.ratio, multiplier=1, nonlinearity=torch.sigmoid)

        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, nclass)

        self.s1=None
        self.s2=None


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        num_nodes = x.shape[1]

        if self.edge_dim > 1:
            edge_attr = self.edge_lin(edge_attr)
            edge_attr = self.edge_act(edge_attr)

        batch = []
        pos = []
        for i in range(len(data.y)):
            batch += [i] * num_nodes
            pos.append(torch.eye(num_nodes))
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        pos = torch.cat(pos).to(device)

        x = self.conv1(x, edge_index, edge_attr, pos)

        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        edge_attr = edge_attr.squeeze()
        # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        self.s1 = torch.sigmoid(score1).view(x.size(0),-1)
        self.s2 = torch.sigmoid(score2).view(x.size(0),-1)

        return x#,self.pool1.weight,self.pool2.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def loss(self, pred, label):
        lamb0 = 1
        lamb1 = 0.1
        lamb2 = 0.1
        lamb3 = 0.1
        lamb4 = 0.1
        lamb5 = 0.1
        criterion = nn.CrossEntropyLoss()
        loss_c = criterion(pred, label)
        loss_p1 = (torch.norm(self.pool1.select.weight, p=2) - 1) ** 2
        loss_p2 = (torch.norm(self.pool2.select.weight, p=2) - 1) ** 2
        loss_tpk1 = self.topk_loss(self.s1, self.ratio)
        loss_tpk2 = self.topk_loss(self.s2, self.ratio)
        loss_consist = 0
        loss = lamb0 * loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 + lamb3 * loss_tpk1 + lamb4 * loss_tpk2 + lamb5 * loss_consist
        return loss

    def topk_loss(self, s, ratio, EPS=1e-10):
        if ratio > 0.5:
            ratio = 1 - ratio
        s = s.sort(dim=1).values
        res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(
            1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
        return res
