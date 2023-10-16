import torch
import os
import numpy as np

class CROHMEDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, args):
        super().__init__()
        self.data_type = data_type
        self.data_path = os.path.join(args.root_path, self.data_type)
        self.data_list = os.listdir(self.data_path)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_path, self.data_list[index]))
        strokes_emb = data['strokes_emb']
        edges_emb = data['edges_emb']
        stroke_labels = data['stroke_labels']
        edge_labels = data['edge_labels']
        los = data['los']
        return strokes_emb, edges_emb, los, stroke_labels, edge_labels
    
    def __len__(self):
        return len(self.data_list)