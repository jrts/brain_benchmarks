import torch
import torch.utils.data
import time
import numpy as np
import csv
import dgl
from tqdm import tqdm
import random
random.seed(42)
from itertools import combinations


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def deriv_m1(X, W):
    return W - X.T @ X
    # return X.T @ (X @ W - X)

def loss_func_m1(X, W, C, l):
    # return 0.5 * np.linalg.norm(X - X @ W, ord='fro') ** 2 + l * np.linalg.norm(np.multiply(C, W), ord=1)
    return 0.5 * np.linalg.norm(W - X.T @ X, ord='fro') ** 2 + l * np.linalg.norm(np.multiply(C, W), ord=1)

def deriv_m2(X, W, L, l1):
    return W - X.T @ X + l1 * W @ L
    # return X.T @ (X @ W - X) + l1 * W @ L

def loss_func_m2(X, W, L, l1, l2):
    # return 0.5 * np.linalg.norm(X - X @ W, ord='fro') ** 2 + 0.5 * l1 * np.trace(W @ L @ W.T) + l2 * np.linalg.norm(W, ord=1)
    return 0.5 * np.linalg.norm(W - X.T @ X, ord='fro') ** 2 + 0.5 * l1 * np.trace(W @ L @ W.T) + l2 * np.linalg.norm(W, ord=1)

def generate_m2_embeddings(ts_data, L, n_nodes, lambda1, lambda2):

    Ws = []
    lr = 1
    n_iters = 100

    zeros = np.zeros((n_nodes, n_nodes))

    for ts_mat in ts_data:

        if ts_mat.shape[0] == n_nodes:
            X = ts_mat.T.copy()
        else:
            X = ts_mat.copy()
        W = np.zeros((n_nodes, n_nodes)).astype(np.double)

        best_loss = float('inf')
        best_w = None
        loss = None
        for i in range(n_iters):
            gradients = deriv_m2(X, W, L, lambda1)
            W -= lr * gradients
            W = np.sign(W) * np.maximum(np.abs(W) - lambda2, zeros)

            loss = loss_func_m2(X, W, L, lambda1, lambda2)

            if loss < best_loss:
                best_loss = loss
                best_w = W

        best_w[best_w > 0] = best_w[best_w > 0] / np.max(best_w[best_w > 0])
        if np.sum(best_w < 0) > 0:
            best_w[best_w < 0] = -best_w[best_w < 0] / np.min(best_w[best_w < 0])
        Ws.append(best_w)

    return np.array(Ws)


def load_distances(n_nodes, parcellation):
    edge_combos = list(combinations(range(n_nodes), 2))
    edge_combos_dict = {}
    for i, e in enumerate(edge_combos):
        edge_combos_dict[e] = i

    dists_mat = np.zeros((n_nodes, n_nodes))
    if parcellation == 'schaefer100':
        file = f'../coordinates/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
        coords_mat = np.loadtxt(file, usecols=(2,3,4), delimiter=',', skiprows=1)
        roi_descs = np.loadtxt(file, dtype=str, usecols=(1), delimiter=',', skiprows=1)
    elif parcellation == 'aal116':
        file = f'../coordinates/aal116_coordinates.csv'
        coords_mat = np.loadtxt(file, usecols=(1,2,3), delimiter=',')
        roi_descs = np.loadtxt(file, dtype=str, usecols=(0), delimiter=',')

    for e in edge_combos:
        dist = np.linalg.norm(coords_mat[e[0],] - coords_mat[e[1],])
        dists_mat[e] = dist
        dists_mat[e[1], e[0]] = dist

    dists_array = np.array([dists_mat[e] for e in edge_combos])

    return coords_mat, edge_combos, edge_combos_dict, dists_mat, dists_array, roi_descs


class PSCRDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_base_url, splits_base_url, dataset_name, threshold):
        t0 = time.time()
        self.name = dataset_name

        parcellation = dataset_name.split('_')[1]
        self.num_nodes = int(parcellation[-3:])

        coords_mat, edge_combos, edge_combos_dict, dists_mat, dists_array, roi_descs = load_distances(self.num_nodes, parcellation)

        dataset_url = f'../bin_datasets/{dataset_name}.bin'
        bin_ds = dgl.load_graphs(dataset_url)

        ts_data = [g.ndata['N_features'].type(torch.double).numpy() for g in bin_ds[0]]
        labels = bin_ds[1]['glabel'].numpy()

        # C = dists_mat / np.max(dists_mat)
        S = np.exp(-dists_mat)
        # S = np.divide(S.T, np.sum(S, axis=1)).T
        D = np.zeros((self.num_nodes, self.num_nodes))
        np.fill_diagonal(D, S.sum(1))
        L = D - S

        matrices = generate_m2_embeddings(ts_data, L, self.num_nodes, 32, 32)

        print("[!] Dataset: ", self.name)

        data = []
        for mat, label in zip(matrices, labels):

            if threshold == 'all':

                medium = np.percentile(mat, 50)
                np.fill_diagonal(mat, medium)

                lower = np.percentile(mat, 20)
                upper = np.percentile(mat, 80)

                if lower == 0.0 or upper == 0.0:
                    e_indices = np.where(np.abs(mat) > 0)
                else:
                    e_indices = np.where(((mat >= lower) & (mat < upper)))
                mat[e_indices] = 0

                e_indices = np.where(mat >= -5)
                selected_edges = np.row_stack(e_indices)

            else:
                medium = np.percentile(mat, 50)
                np.fill_diagonal(mat, medium)

                l, u = threshold.split('_')[:2]

                lower = np.percentile(mat, int(l))
                upper = np.percentile(mat, int(u))

                if lower == 0.0 or upper == 0.0:
                    e_indices = np.where(np.abs(mat) > 0)
                else:
                    e_indices = np.where(((mat < lower) | (mat >= upper)))

                selected_edges = np.row_stack(e_indices)
                self_loop_indices = np.arange(self.num_nodes)
                selected_edges = np.concatenate((selected_edges, [self_loop_indices, self_loop_indices]), axis=1)

                np.fill_diagonal(mat, 0)

            g = dgl.graph(data=([], []), num_nodes=self.num_nodes)
            g.add_edges(selected_edges[0], selected_edges[1])
            # if edge_feature:
            g.edata['feat'] = torch.from_numpy(mat[selected_edges[0], selected_edges[1]]).float().unsqueeze(-1)
            g.ndata['feat'] = torch.from_numpy(mat).float()

            data.append([g, label])

        dataset = self.format_dataset(data)
        # this function splits data into train/val/test and returns the indices
        self.all_idx = self.load_splits(splits_base_url, dataset_name)

        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx[split_num][0]]) for split_num in range(10)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx[split_num][1]]) for split_num in range(10)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx[split_num][2]]) for split_num in range(10)]

        self.n_folds = len(self.train)
        
        print("Time taken: {:.4f}s".format(time.time()-t0))


    def load_splits(self, splits_base_url, dataset):
        splits = []

        ds_split = dataset.split('_')
        for section in ['train', 'val', 'test']:
            with open(f'{splits_base_url}/{ds_split[0]}_{ds_split[2]}/{section}.index', 'r') as f:
                reader = csv.reader(f)
                splits.append([list(map(int, idx)) for idx in reader])
                f.close()
        
        return [[train, val, test] for train, val, test, in zip(splits[0], splits[1], splits[2])]


    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        # for graph in graphs:
        #     # adding edge features for Residual Gated ConvNet, if not there
        #     if 'feat' not in graph.edata.keys():
        #         edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
        #         graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

        return DGLFormDataset(graphs, labels)
    
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)
        
        in_dim = g.ndata['feat'].shape[1]
        
        # use node feats to prepare adj
        adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
        adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
        
        for node, node_feat in enumerate(g.ndata['feat']):
            adj_node_feat[1:, node, node] = node_feat

        x_node_feat = adj_node_feat.unsqueeze(0)
        
        return x_node_feat, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    # def _add_self_loops(self):

    #     # function for adding self loops
    #     # this function will be called only if self_loop flag is True
    #     for split_num in range(10):
    #         self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
    #         self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
    #         self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
    #     for split_num in range(10):
    #         self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
    #         self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
    #         self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)
