import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from dgl.nn import GraphConv, AvgPooling, MaxPooling, NNConv
from pooling.topkpool import TopKPooling
from pooling.sagpool import SAGPool

class LINet(torch.nn.Module):
    def __init__(self, net_params, poolmethod='sag'):
        super(LINet, self).__init__()

        indim = net_params['in_dim']
        in_dim = net_params['hidden_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        dropout = net_params['dropout']

        self.lamb0 = 1  # classification loss weight
        self.lamb1 = 1  # s1 unit regularization
        self.lamb2 = 1  # s2 unit regularization
        self.lamb3 = 1  # s1 distance regularization
        self.lamb4 = 1  # s2 distance regularization
        self.lamb5 = 0  # s1 consistence regularization
        self.lamb6 = 0  # s2 consistence regularization

        self.indim = indim
        self.ratio = net_params['pool_ratio']
        self.poolmethod = poolmethod
        self.gap = AvgPooling()
        self.gmp = MaxPooling()

        self.edge_dim = net_params['edge_dim']

        n1 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, in_dim * indim))
        self.conv1 = NNConv(indim, in_dim, n1, 'sum')
        if self.poolmethod == 'topk':
            self.pool1 = TopKPooling
        elif self.poolmethod == 'sag':
            self.pool1 = SAGPool(in_dim, ratio=self.ratio)  # 0.4 data1 10 fold

        n2 = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, in_dim * hidden_dim))
        self.conv2 = NNConv(in_dim, hidden_dim, n2, 'sum')
        if self.poolmethod == 'topk':
            self.pool2 = TopKPooling
        elif self.poolmethod == 'sag':
            self.pool2 = SAGPool(hidden_dim, ratio=self.ratio)

        self.fc1 = torch.nn.Linear(hidden_dim * 4, out_dim)
        self.bn4 = torch.nn.BatchNorm1d(out_dim)
        self.fc2 = torch.nn.Linear(out_dim, hidden_dim)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, net_params['n_classes'])

    def forward(self, g, h, e):

        h = self.conv1(g, h, e).flatten(1)  # h: [4000, 17*8]
        g, h, _, score1, e = self.pool1(g, h, e)  # h: [2000, 17*8]
        g1 = torch.cat([self.gap(g, h), self.gmp(g, h)], dim=-1)  # g1: [20, 17*8*2]

        h = self.conv2(g, h, e).flatten(1)  # h: [2000, 17*8]
        g, h, _, score2, e = self.pool2(g, h, e)  # h: [1000, 17*8]
        g2 = torch.cat([self.gap(g, h), self.gmp(g, h)], dim=-1)  # g2: [20, 17*8*2]

        x = torch.cat([g1, g2], dim=1)  # concate. x: [20, 17*8*2*2]

        x = self.bn4(F.relu(self.fc1(x))) if x.size(0) != 1 else F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn5(F.relu(self.fc2(x))) if x.size(0) != 1 else F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x, score1, score2

    def loss(self, pred, label, s1, s2):
        loss_c = F.nll_loss(pred, label)
        loss_dist1 = self.dist_loss(s1, self.ratio)
        loss_dist2 = self.dist_loss(s2, self.ratio)
        # loss_consist = self.consist_loss(s1[label == 1]) + self.consist_loss(s1[label == 0])
        loss = self.lamb0 * loss_c \
               + self.lamb3 * loss_dist1 + self.lamb4 * loss_dist2  # + self.lamb5 * loss_consist
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(pred, label)
        return loss

    def dist_loss(self, s, ratio):
        s = s.sort().values
        EPS = 1e-15
        ratio = 1 - ratio
        s = s.sort().values
        res = -torch.log(s[-int(len(s)*ratio):]+EPS).mean() -torch.log(1-s[:int(len(s)*ratio)]+EPS).mean()
        return res

    # def consist_loss(self, s):
    #     if len(s) == 0:
    #         return 0
    #     else:
    #         s = torch.sigmoid(s)
    #         W = torch.ones(s.shape[0],s.shape[0])
    #         D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    #         L = D-W
    #         # L = L.to(device)
    #         res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    #         return res
