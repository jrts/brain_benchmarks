import numpy as np
import torch
from itertools import combinations

def load_coords(parcellation, n_nodes, self_loop=True):
    if parcellation == 'schaefer100':
        coordinates_file = f'../coordinates/schaefer100_coordinates.csv'
        coords = np.loadtxt(coordinates_file, usecols=(2,3,4), delimiter=',', skiprows=1)
    elif parcellation == 'aal116':
        coordinates_file = f'../coordinates/aal116_coordinates.csv'
        coords = np.loadtxt(coordinates_file, usecols=(1,2,3), delimiter=',')
    else:
        raise NotImplemented
    
    dist_mat = np.zeros((n_nodes, n_nodes))

    if self_loop:
        edge_combos = torch.triu_indices(n_nodes, n_nodes).T.numpy()
    else:
        edge_combos = list(combinations(range(n_nodes), 2))

    edge_combos_dict = {}
    for i, e in enumerate(edge_combos):
        if not isinstance(e, tuple): e = tuple(e)
        edge_combos_dict[e] = i
        dist = np.linalg.norm(coords[e[0]] - coords[e[1]])
        dist_mat[e] = dist
        dist_mat[e[1], e[0]] = dist
    return dist_mat, coords, edge_combos, edge_combos_dict