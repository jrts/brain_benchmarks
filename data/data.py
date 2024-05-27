"""
    File to load dataset based on user control from main file
"""
from data.dgl_dataset import DGLDataset
from data.pyg_dataset import PYGDataset
from data.pscr_dataset import PSCRDataset


def LoadData(dataset_base_url, splits_base_url, dataset_name, model_name, threshold):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    if model_name in ['BrainGNN', 'BNTF']:
        return PYGDataset(dataset_base_url, splits_base_url, dataset_name, threshold)
    if model_name in ['PSCR']:
        return PSCRDataset(dataset_base_url, splits_base_url, dataset_name, threshold)
    else:
        return DGLDataset(dataset_base_url, splits_base_url, dataset_name, threshold)
