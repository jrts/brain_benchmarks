import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import csv


class PYGDataset(InMemoryDataset):

    def __init__(self, root, splits_base_url, filename, dataset_suffix, transform=None, pre_transform=None, pre_filter=None):
        self.name = filename
        self.dataset_suffix = dataset_suffix
        self.cv_splits = None
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.num_nodes = self.data.num_nodes
        self.cv_splits = self.load_splits(splits_base_url, filename)

    def num_nodes(self):
        return int(self.name.split('_')[1][-3:])
    
    def n_classes(self):
        return len(torch.unique(self.data.y))

    def edge_dim(self):
        return 1


    @property
    def processed_file_names(self):
        return [f"{self.name.split('.')[0]}_{self.dataset_suffix}_processed.pt"]
    

    def process(self):
        dataset = np.load(f'{self.root}/{self.name}.npy', allow_pickle=True).item()
        matrices = dataset['connectivity_matrices']
        labels = dataset['labels']

        num_nodes = matrices.shape[1]
        data_list = []
        for idx, (mat, label) in enumerate(zip(matrices, labels)):
            # make sure the diagonal nodes are not included
            medium = np.percentile(mat, 50)
            np.fill_diagonal(mat, medium)

            l, u = self.dataset_suffix.split('_')[:2]

            lower = np.percentile(mat, int(l))
            upper = np.percentile(mat, int(u))

            if lower == 0.0 or upper == 0.0:
                e_indices = np.where(np.abs(mat) > 0)
            else:
                e_indices = np.where(((mat < lower) | (mat >= upper)))

            selected_edges = np.row_stack(e_indices)
            selected_edges = selected_edges[:, selected_edges[0] < selected_edges[1]]

            self_loop_indices = np.arange(num_nodes)
            selected_edges = np.concatenate((selected_edges, [selected_edges[1], selected_edges[0]], [self_loop_indices, self_loop_indices]), axis=1)

            np.fill_diagonal(mat, 0.0)

            num_edges = selected_edges.shape[1]
            edge_attr = mat[selected_edges[0], selected_edges[1]].reshape((-1, 1))
                    
            data = Data(
                x=torch.FloatTensor(mat), 
                y=torch.LongTensor([label]), 
                edge_index=torch.LongTensor(selected_edges), 
                edge_attr=torch.FloatTensor(edge_attr),
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        data.num_nodes = num_nodes
        data.num_edges = num_edges
        data.edge_dim = edge_attr.shape[1]
        data.n_classes = len(np.unique(labels))
        torch.save((data, slices), self.processed_paths[0])


    def _process(self):
        if os.path.exists(self.processed_paths[0]):
            return
        
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        self.process()


    def load_splits(self, splits_base_url, dataset):
        splits = []

        ds_split = dataset.split('_')
        for section in ['train', 'val', 'test']:
            with open(f'{splits_base_url}/{ds_split[0]}_{ds_split[2]}/{section}.index', 'r') as f:
                reader = csv.reader(f)
                splits.append([list(map(int, idx)) for idx in reader])
                f.close()
        
        return [[train, val, test] for train, val, test, in zip(splits[0], splits[1], splits[2])]