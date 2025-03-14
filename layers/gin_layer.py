import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from norm.GraphNorm import Norm

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """
    def __init__(self, apply_func, aggr_type, dropout, batch_norm, residual=False, init_eps=0, learn_eps=False, e_feat=False):
        super().__init__()
        self.apply_func = apply_func

        self.message_func = fn.copy_u('h', 'm') if not e_feat else fn.u_mul_e('h', 'e', 'm')
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
            
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
        
        if in_dim != out_dim:
            self.residual = False
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        # self.batchnorm_h = nn.BatchNorm1d(out_dim)
        # self.batchnorm_e = nn.BatchNorm1d(out_dim)
        self.batchnorm_h = Norm('gn', out_dim)
        self.batchnorm_e = Norm('bn', out_dim)

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        e_in = e
        
        g = g.local_var()
        g.ndata['h'] = h
        g.edata['e'] = e
        g.update_all(self.message_func, self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        e = g.edata['e']

        if self.apply_func is not None:
            h = self.apply_func(h)
            e = self.apply_func(e)

        if self.batch_norm:
            h = self.batchnorm_h(g, h)  # batch normalization
            e = self.batchnorm_e(g, e)
       
        h = F.relu(h)  # non-linear activation
        e = F.relu(e)
        
        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    
class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
