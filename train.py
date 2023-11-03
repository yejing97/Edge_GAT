from Model.LitModel import LitModel
from Dataset.Datamodule import CROHMEDatamodule
from Preprocessing.make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
import sys

root_path = sys.path[0]
results_path = os.path.join(root_path, 'val_results')
parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='/home/xie-y/data/Edge_GAT/')
# parser.add_argument('--ckpt_path', type=str, default='/home/xie-y/Edge_GAT/pretrain_logs/S100_R5_Speed_False_lr_0.001/version_2/checkpoints/epoch=26-step=70470.ckpt')
# parser.add_argument('--results_path', type=str, default='/home/xie-y/Edge_GAT/val_results/')
parser.add_argument('--stroke_emb_nb', type=int, default=100)
parser.add_argument('--rel_emb_nb', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--lambda1', type=float, default=0.8)
parser.add_argument('--lambda2', type=float, default=0.6)

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--accelerator', type=str, default="gpu")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
# device = 'gpu' if torch.cuda.is_available() else 'cpu'
if args.device == 'cuda':
    data_path = '/home/xie-y/data/Edge_GAT/'
else:
    data_path = '/home/e19b516g/yejing/data/data_for_graph/'
npz_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed)
npz_path = os.path.join(data_path, npz_name)
if not os.path.exists(npz_path):
    os.makedirs(npz_path)
    make_data(os.path.join(data_path, 'INKML'), npz_path, args.stroke_emb_nb, args.rel_emb_nb, args.speed)

exp_name = 'lr_' + str(args.lr) + '_bs_' + str(args.batch_size) + '_epoch_' + str(args.epoch) + '_dropout_' + str(args.dropout) + '_l1_' + str(args.lambda1) + '_l2_' + str(args.lambda2)
logger_path = os.path.join(root_path, 'logs' , npz_name)
logger = TensorBoardLogger(save_dir=logger_path, name=exp_name)
val_results_path = os.path.join(results_path, npz_name, exp_name)
if not os.path.exists(val_results_path):
    os.makedirs(val_results_path)
    version = 0
else:
    version = len(os.listdir(val_results_path))
val_results_path = os.path.join(val_results_path, 'version_' + str(version))
os.makedirs(val_results_path)

dm = CROHMEDatamodule(
    root_path = npz_path,
    shuffle = args.shuffle,
    batch_size = args.batch_size,
    num_workers = args.num_workers,
    reload_dataloaders_every_n_epochs = 1
)

model = LitModel(
    node_input_size = args.stroke_emb_nb,
    edge_input_size = args.rel_emb_nb * 4,
    gat_input_size = 114,
    gat_hidden_size = 512,
    gat_output_size = 128,
    gat_n_heads = 8,
    node_class_nb = 114,
    edge_class_nb = 14,
    # ckpt_path = args.ckpt_path,
    results_path = val_results_path,
    dropout = args.dropout,
    lambda1 = args.lambda1,
    lambda2 =  args.lambda2,
    lr = args.lr,
    device = args.device
)


trainer = pl.Trainer(max_epochs=args.epoch, accelerator=args.device, devices=1, logger=logger)
trainer.fit(model.to(args.device), dm)