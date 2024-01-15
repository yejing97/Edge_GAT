import pytorch_lightning as pl
import torch
import random
import yaml
from Dataset.Dataset import CROHMEDataset

class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(self, npz_path, config_path) -> None:
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.root_path = npz_path
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        self.batch_size = config['batch_size']
        self.max_node = config['max_node']
        self.am_type = config['am_type']
        # self.node_norm = config['node_norm']
        self.node_type = config['node_type']
        self.max_padding_size = int(self.max_node//2)
        # self.train = 'train'
        # self.val = 'val'
        # self.test = 'test'
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        # self.random_padding_size = random.randint(0, 7)
        self.random_padding_size = random.randint(0, self.max_padding_size)
        # print('random_padding_size: ', self.random_padding_size)
        # self.dataset_train = CROHMEDataset('train', self.root_path, self.batch_size, self.max_node, self.random_padding_size)
        # self.dataset_val = CROHMEDataset('val', self.root_path, self.batch_size, self.max_node, self.random_padding_size)
        # self.dataset_test = CROHMEDataset('test', self.root_path, self.batch_size, self.max_node, self.random_padding_size)

    def train_dataloader(self):
        # self.setup('fit')
        
        self.random_padding_size = random.randint(0, self.max_padding_size)
        self.dataset_train = CROHMEDataset('train', self.root_path, self.batch_size, self.max_node, self.random_padding_size, self.am_type, self.node_type)
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = self.batch_size, 
            # batch_size = 1,
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    
    def val_dataloader(self):
        # self.setup('fit')
        # self.random_padding_size = random.randint(0, 3)
        self.dataset_val = CROHMEDataset('val', self.root_path, self.batch_size, self.max_node, self.random_padding_size, self.am_type, self.node_type)
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            # batch_size = self.batch_size,
            batch_size = self.batch_size,
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    
    def test_dataloader(self):
        # self.setup()
        self.dataset_test = CROHMEDataset('val', self.root_path, self.batch_size, -1, self.random_padding_size, self.am_type, self.node_type)
        return torch.utils.data.DataLoader(
            self.dataset_test, 
            batch_size = 1, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    