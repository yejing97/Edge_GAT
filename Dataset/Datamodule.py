import pytorch_lightning as pl
import torch
import random
from Dataset.Dataset import CROHMEDataset

class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(self, **args) -> None:
        super().__init__()
        self.root_path = args['root_path']
        self.shuffle = args['shuffle']
        self.num_workers = args['num_workers']
        self.batch_size = args['batch_size']
        self.max_node = args['max_node']
        # self.train = 'train'
        # self.val = 'val'
        # self.test = 'test'
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        # self.random_padding_size = random.randint(0, 7)
        self.random_padding_size = 0
        print('random_padding_size: ', self.random_padding_size)
        self.dataset_train = CROHMEDataset('train', self.root_path, self.batch_size, self.max_node, self.random_padding_size)
        self.dataset_val = CROHMEDataset('val', self.root_path, self.batch_size, self.max_node, self.random_padding_size)
        self.dataset_test = CROHMEDataset('test', self.root_path, self.batch_size, self.max_node, self.random_padding_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = self.batch_size, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size = self.batch_size,
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, 
            batch_size = self.batch_size, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    