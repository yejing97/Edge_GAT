import torch
import os
import numpy as np

class CROHMEDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, root_path):
        super().__init__()
        self.data_type = data_type
        self.data_path = os.path.join(root_path, self.data_type)
        self.data_list = os.listdir(self.data_path)[:100]

    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_path, self.data_list[index]))
        strokes_emb = torch.from_numpy(data['strokes_emb']).float()
        edges_emb = torch.from_numpy(data['edges_emb']).float()
        stroke_labels = torch.from_numpy(data['stroke_labels']).long()
        edge_labels = torch.from_numpy(data['edge_labels']).long()
        los = torch.from_numpy(data['los']).long()
        return strokes_emb, edges_emb.squeeze().reshape(edges_emb.shape[0], edges_emb.shape[1], edges_emb.shape[2]*edges_emb.shape[3]), los, stroke_labels, edge_labels
    
    def __len__(self):
        return len(self.data_list)