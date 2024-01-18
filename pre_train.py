from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tsai.all import *
import torch
import os
import argparse
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/S100_R5_Speed_False/')
parser.add_argument('--root_path', type=str, default='/home/xie-y/data/Edge_GAT/S150_R10/')
parser.add_argument('--stroke_emb_nb', type=int, default=100)
parser.add_argument('--stroke_class_nb', type=int, default=102)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--min_delta', type=float, default=0.00001)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

def make_pt(npz_path, tgt_path, type):
    X = torch.empty(0, 100, 2)
    y = torch.empty(0)
    for root, _, files in os.walk(os.path.join(npz_path, type)):
        for file in files:
            if file.endswith('.npz'):
                data = np.load(os.path.join(root, file))
                strokes_emb = torch.from_numpy(data['strokes_emb']).float()
                stroke_labels = torch.from_numpy(data['stroke_labels']).long()
                X = torch.cat((X, strokes_emb), dim=0)
                y = torch.cat((y, stroke_labels), dim=0)
    torch.save(X, os.path.join(tgt_path, type + '_X.pt'))
    torch.save(y, os.path.join(tgt_path, type + '_y.pt'))
    print('make pt ' + type + ' done')

# load data


# train_ds = TSDatasets(train_X, train_y)
# val_ds = TSDatasets(val_X, val_y)
# dl = TSDataLoaders.from_dsets(train_ds, val_ds, bs=args.batch_size, num_workers=0)

class PretrainDatamodule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
    
    def setup(self, stage: str) -> None:
        if os.path.isfile(os.path.join(args.root_path, 'train_X.pt')) and os.path.isfile(os.path.join(args.root_path, 'train_y.pt')):
            print(os.path.join(args.root_path, 'train_X.pt'))
            print('train pt for pre train exists')
        else:
            make_pt(args.root_path, args.root_path, 'train')
        train_X = torch.load(os.path.join(args.root_path, 'train_X.pt'))
        train_y = torch.load(os.path.join(args.root_path, 'train_y.pt'))
        if os.path.isfile(os.path.join(args.root_path, 'val_X.pt')) and os.path.isfile(os.path.join(args.root_path, 'val_y.pt')):
            print('val pt for pre train exists')
        else:
            make_pt(args.root_path, args.root_path, 'val')
        val_X = torch.load(os.path.join(args.root_path, 'val_X.pt'))
        val_y = torch.load(os.path.join(args.root_path, 'val_y.pt'))
        if os.path.isfile(os.path.join(args.root_path, 'test_X.pt')) and os.path.isfile(os.path.join(args.root_path, 'test_y.pt')):
            print('test pt for pre train exists')
        else:
            make_pt(args.root_path, args.root_path, 'test')
        test_X = torch.load(os.path.join(args.root_path, 'test_X.pt'))
        test_y = torch.load(os.path.join(args.root_path, 'test_y.pt'))
        # return super().setup(stage)
        self.dataset_train = TSDatasets(train_X, train_y)
        self.dataset_val = TSDatasets(val_X, val_y)
        self.dataset_test = TSDatasets(test_X, test_y)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, 
            batch_size = args.batch_size, 
            shuffle = args.shuffle, 
            num_workers = args.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size = args.batch_size, 
            shuffle = args.shuffle, 
            num_workers = args.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, 
            batch_size = args.batch_size, 
            shuffle = args.shuffle, 
            num_workers = args.num_workers)
    
class LightModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = XceptionTime(args.stroke_emb_nb, args.stroke_class_nb)
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        strokes_emb, strokes_label = batch
        output = self.model(strokes_emb.to(args.device))
        loss = self.loss(output, strokes_label.to(args.device).long())
        self.log('train_loss', loss)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, strokes_label = batch
        output = self.model(strokes_emb.to(args.device))
        loss = self.loss(output, strokes_label.to(args.device).long())
        acc = accuracy_score(strokes_label.cpu().numpy(), output.cpu().argmax(dim=1).numpy())
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=args.min_delta, verbose=True),
            'monitor': 'val_acc',
            'frequency': self.trainer.check_val_every_n_epoch
        }
# load model

model = LightModel()
dm = PretrainDatamodule()
early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")
exp_name = args.root_path.split('/')[-2] + '_lr_' + str(args.lr) + '_batch_size_' + str(args.batch_size)
logger = pl.loggers.TensorBoardLogger('pretrain_logs', name=exp_name)
trainer = pl.Trainer(
    max_epochs=args.epoch,
    accelerator="auto",
    gpus= 1,
    callbacks=[early_stop_callback],
    logger=logger
)
trainer.fit(model.to(args.device), dm)