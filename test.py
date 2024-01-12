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
parser.add_argument('--checkpoint_name', type=str, default = 'test1')
 
args = parser.parse_args()

if __name__ == '__main__':
            
        root_path = sys.path[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_path = os.path.join(root_path, 'config')
    
        if torch.cuda.is_available():
            data_path = '/home/xie-y/data/Edge_GAT/'
            checkpoint_path = '/home/xie-y/data/Edge_GAT/checkpoints/'
        else:
            data_path = '/home/e19b516g/yejing/data/data_for_graph/'
            checkpoint_path = '/home/e19b516g/yejing/data/data_for_graph/checkpoints/'

        stroke_emb_nb = 150
        rel_emb_nb = 40
        # npz_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed) + '_' + str(args.norm)
        npz_name = 'S'+ str(stroke_emb_nb) + '_R' + str(rel_emb_nb)
    
        npz_path = os.path.join(data_path, npz_name)
    
    
        dm = CROHMEDatamodule(
            npz_path, 
            # config,
            )
        # dm.setup('fit')
        model = LitModel.load_from_checkpoint(
            os.path.join(checkpoint_path, args['checkpoint_name'] + '.ckpt'),
            )
        
        trainer = pl.Trainer(
            gpus = 1,
            accelerator="dp",
            )
        trainer.test(model, datamodule=dm)


