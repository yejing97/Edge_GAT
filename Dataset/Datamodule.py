import pytorch_lightning as pl
import torch
from Dataset import CROHMEDataset

class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # self.train = 'train'
        # self.val = 'val'
        # self.test = 'test'
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        self.dataset_train = CROHMEDataset('train', self.args)
        self.dataset_val = CROHMEDataset('val', self.args)
        self.dataset_test = CROHMEDataset('test', self.args)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = 1, 
            shuffle = True, 
            num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size = 1, 
            shuffle = False, 
            num_workers=self.args.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, 
            batch_size = 1, 
            shuffle = False, 
            num_workers=self.args.num_workers)