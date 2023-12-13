from typing import Any
import pytorch_lightning as pl
import torch
import random
from Dataset.Dataset import CROHMEDataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F



class Seg_DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, shuffle, num_workers, node_norm, max_node) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.node_norm = node_norm
        self.am_type = 'seg'
        self.max_node = max_node
        self.max_padding_size = int(self.max_node//2)

    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage: str):
        self.random_padding_size = random.randint(0, self.max_padding_size)

    def train_dataloader(self):
        self.dataset_train = CROHMEDataset('train', self.data_path, self.batch_size, self.max_node, self.random_padding_size, self.am_type, self.node_norm)
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = self.batch_size, 
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )

    def val_dataloader(self):
        self.dataset_val = CROHMEDataset('val', self.data_path, self.batch_size, self.max_node, self.random_padding_size, self.am_type, self.node_norm)
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size = self.batch_size,
            shuffle = self.shuffle, 
            num_workers=self.num_workers
            )
    
class Seg_LitModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.embedding = torch.nn.Embedding(300, 128)
        self.bi_gru = torch.nn.GRU(300, 32, 2, bidirectional=True)
        self.conc = torch.nn.Linear(64, 2)
        self.softmax = torch.nn.Softmax(dim=2)
        # binary cross entropy loss
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # x = self.embedding(x)
        x,_ = self.bi_gru(x.reshape(x.shape[0], x.shape[1], -1))
        x = self.conc(x)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.to('cuda'))
        y_pred = y_hat[:, :, 0].to('cuda')
        loss = F.binary_cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.to('cuda'))
        y_pred = y_hat[:, :, 0].to('cuda')
        loss = F.binary_cross_entropy(y_pred, y)
        # acc = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        acc = accuracy_score(y.cpu().numpy(), y_hat.cpu().argmax(dim=2).numpy())
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

model = Seg_LitModel()
dm = Seg_DataModule('/home/e19b516g/yejing/data/data_for_graph/S150_R10', 128, True, 0, True, 8)
logger = pl.loggers.TensorBoardLogger('gru_logs', name='seg')
trainer = pl.Trainer(max_epochs=10, logger=logger, accelerator="auto", devices=1, reload_dataloaders_every_n_epochs=1)
trainer.fit(model.to('cuda'), dm)
    
