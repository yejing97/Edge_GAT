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
parser.add_argument('--stroke_emb_nb', type=int, default=150)
parser.add_argument('--stroke_class_nb', type=int, default=102)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--min_delta', type=float, default=0.00001)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


def connected_components_mask(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.cpu()
            # batch_nb = adjacency_matrix.shape[0]
            num_nodes = adjacency_matrix.shape[0]
            visited = torch.zeros(num_nodes, dtype=bool)
            def dfs(node, component):
                visited[node] = True
                component.append(node)
                for neighbor in range(num_nodes):
                    if adjacency_matrix[node][neighbor] == 1 and not visited[neighbor]:
                        dfs(neighbor, component)
            components = []
            for node in range(num_nodes):
                if not visited[node]:
                    component = []
                    dfs(node, component)
                    components.append(component)
            max_node = max(max(component) for component in components) + 1
            mask = torch.zeros(len(components), max_node, dtype=torch.int)
            for i, component in enumerate(components):
                for node in component:
                    mask[i][node] = 1
            return mask, components
    
def sub_graph_pooling(node_feat, mask):
    node_feat_repeat = node_feat.unsqueeze(0).repeat(mask.shape[0], 1,1,1)
    node_out = torch.sum(node_feat_repeat * mask.unsqueeze(-1).unsqueeze(-1), dim=1)
    node_out = node_out/mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
    node_out = torch.matmul(mask.t().float(), node_out.reshape(node_out.shape[0], -1))
    node_out = node_out.reshape(node_feat.shape[0], node_feat.shape[1], -1)
    return node_out

def make_pt(npz_path, tgt_path, type):
    X = torch.empty(0, 150, 2)
    y = torch.empty(0)
    for root, dirs, files in os.walk(os.path.join(npz_path, type, 'strokes_emb')):
        for file in files:
            if file.endswith('.npy'):
                print(file)
                # data = np.load(os.path.join(root, file))
                strokes_emb = np.load(os.path.join(root, file))
                strokes_emb = torch.from_numpy(strokes_emb).long()
                stroke_labels = np.load(os.path.join(npz_path, type, 'stroke_labels', file))
                stroke_labels = torch.from_numpy(stroke_labels).long()
                edge_labels = np.load(os.path.join(npz_path, type, 'edge_labels', file))
                edge_labels = torch.from_numpy(edge_labels).long()
                am = torch.where(edge_labels == 1, 1, 0)
                try:
                    sym_mask,_ = connected_components_mask(am)
                    strokes_emb = sub_graph_pooling(strokes_emb, sym_mask)
                except:
                    print(file + ' has error')

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
            num_workers = args.num_workers
            )

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
        # self.linear = torch.nn.Linear(384, args.stroke_class_nb)
        self.softmax = torch.nn.Softmax(dim = -1)
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        strokes_emb, strokes_label = batch
        output = self.model(strokes_emb.to(args.device))
        # output = self.linear(output)
        output = self.softmax(output)
        loss = self.loss(output, strokes_label.to(args.device).long())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, strokes_label = batch
        output = self.model(strokes_emb.to(args.device))
        # output = self.linear(output)
        output = self.softmax(output)
        loss = self.loss(output, strokes_label.to(args.device).long())
        acc = accuracy_score(strokes_label.cpu().numpy(), output.cpu().argmax(dim=1).numpy())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
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
early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
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