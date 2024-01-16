from Model.LitModel import LitModel
# from Model.MOModel import MOModel
from Dataset.Datamodule import CROHMEDatamodule
# from make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
import sys
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_name', type=str, default = 'edge_class14-epoch=99-val_acc_node=0.86')
parser.add_argument('--config_name', type=str, default='edge_class14')
 
args = parser.parse_args()
args = vars(parser.parse_args())


if __name__ == '__main__':
            
        root_path = sys.path[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_path = os.path.join(root_path, 'config', args['config_name'] + '.yaml')
    
        if torch.cuda.is_available():
            data_path = '/home/xie-y/data/Edge_GAT/'
            checkpoint_path = '../checkpoints/'
        else:
            data_path = '/home/e19b516g/yejing/data/data_for_graph/'
            checkpoint_path = '/home/e19b516g/yejing/code/Edge_GAT/checkpoints/'

        stroke_emb_nb = 150
        rel_emb_nb = 40
        # npz_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed) + '_' + str(args.norm)
        npz_name = 'S'+ str(stroke_emb_nb) + '_R10'
    
        npz_path = os.path.join(data_path, npz_name)
    
    
        dm = CROHMEDatamodule(
            npz_path = npz_path, 
            config_path = config_path,
            )
        # dm.setup('fit')
        model = LitModel.load_from_checkpoint(
            os.path.join(checkpoint_path, args['checkpoint_name'] + '.ckpt'),
            config_path = config_path,
            results_path = './results',
            mode = 'train'
            )
        
        trainer = pl.Trainer(
            max_epochs=90,
            gpus = 1,
            accelerator="auto",
            )
        trainer.test(model, datamodule=dm)


