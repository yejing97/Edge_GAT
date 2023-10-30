import pytorch_lightning as pl
import torch
from Dataset.Dataset import CROHMEDataset

class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(self, **args) -> None:
        super().__init__()
        print(args)
        self.root_path = args['root_path']
        self.shuffle = args['shuffle']
        self.num_workers = args['num_workers']
        # self.train = 'train'
        # self.val = 'val'
        # self.test = 'test'
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        self.dataset_train = CROHMEDataset('train', self.root_path)
        self.dataset_val = CROHMEDataset('val', self.root_path)
        self.dataset_test = CROHMEDataset('test', self.root_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = 1, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size = 1, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, 
            batch_size = 1, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers)
    