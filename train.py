from Model.LitModel import LitModel
from Dataset.Datamodule import CROHMEDatamodule
from Preprocessing.make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/')
parser.add_argument('--stroke_emb_nb', type=int, default=100)
parser.add_argument('--rel_emb_nb', type=int, default=5)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--lambda2', type=float, default=0.5)

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--accelerator', type=str, default="gpu")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
# device = 'gpu' if torch.cuda.is_available() else 'cpu'

npz_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed)
npz_path = os.path.join(args.root_path, npz_name)
if not os.path.exists(npz_path):
    os.makedirs(npz_path)
    make_data(os.path.join(args.root_path, 'INKML'), npz_path, args.stroke_emb_nb, args.rel_emb_nb, args.speed)


dm = CROHMEDatamodule(
    root_path = npz_path,
    shuffle = args.shuffle,
    num_workers = args.num_workers
)

model = LitModel(
    node_input_size = 100,
    edge_input_size = 20,
    gat_input_size = 114,
    gat_hidden_size = 64,
    gat_output_size = 128,
    gat_n_heads = 8,
    node_class_nb = 114,
    edge_class_nb = 14,
    ckpt_path = '/home/xie-y/Edge_GAT/pretrain_logs/S100_R5_Speed_False_lr_0.001/version_2/checkpoints/epoch=26-step=70470.ckpt',
    dropout = args.dropout,
    lambda1 = args.lambda1,
    lambda2 =  args.lambda2,
    lr = args.lr,
    device = args.device
)
exp_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed) + '_lr_' + str(args.lr) + '_dropout_' + str(args.dropout) + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2)
logger = TensorBoardLogger('tb_logs', name=exp_name)

trainer = pl.Trainer(max_epochs=args.epoch, accelerator=args.device, devices=1, logger=logger)
trainer.fit(model.to(args.device), dm)