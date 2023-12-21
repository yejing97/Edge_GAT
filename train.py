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
parser.add_argument('--config_name', type=str, default = 'for_new_finetune')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--results_path', type=str, default='val_results')
parser.add_argument('--logs_path', type=str, default='new_logs')

args = parser.parse_args()
args = vars(parser.parse_args())
print(args)

if __name__ == '__main__':
        
    root_path = sys.path[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = os.path.join(root_path, 'config')

    if torch.cuda.is_available():
        data_path = '/home/xie-y/data/Edge_GAT/'
    else:
        data_path = '/home/e19b516g/yejing/data/data_for_graph/'
    config = os.path.join(config_path, args['config_name'] + '.yaml')
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    stroke_emb_nb = cfg['stroke_emb_nb']
    rel_emb_nb = cfg['rel_emb_nb']
    # npz_name = 'S'+ str(args.stroke_emb_nb) + '_R' + str(args.rel_emb_nb) + '_Speed_' + str(args.speed) + '_' + str(args.norm)
    npz_name = 'S'+ str(stroke_emb_nb) + '_' + cfg['edge_feat']

    npz_path = os.path.join(data_path, npz_name)
    # if not os.path.exists(npz_path):
        # os.makedirs(npz_path)

        # make_data(os.path.join(data_path, 'INKML'), npz_path, stroke_emb_nb, rel_emb_nb, args.speed, 'stroke')

    exp_name = args['config_name']
    logger_path = os.path.join(root_path, args['logs_path'] ,args['mode'], npz_name)
    logger = TensorBoardLogger(save_dir=logger_path, name=exp_name)
    results_path = os.path.join(root_path, args['results_path'])
    val_results_path = os.path.join(results_path, npz_name, exp_name)
    if not os.path.exists(val_results_path):
        os.makedirs(val_results_path)
        version = 0
    else:
        version = len(os.listdir(val_results_path))
    val_results_path = os.path.join(val_results_path, 'version_' + str(version))
    os.makedirs(val_results_path)

    dm = CROHMEDatamodule(
        npz_path = npz_path,
        config_path = config
    )

    model = LitModel(
        config_path = config,
        mode = args['mode'],
        results_path = val_results_path
    )
    num_available_gpus = torch.cuda.device_count()

    # for device_idx in range(num_available_gpus):
    #     print(torch.cuda.get_device_name(device_idx))
    #     try:
    #         torch.cuda.set_device(device_idx)
    trainer = pl.Trainer(max_epochs = cfg['epoch'], accelerator="auto",auto_select_gpus=True, gpus= 1,logger=logger,reload_dataloaders_every_n_epochs=cfg['reload_dataloaders_every_n_epochs'])
        # ['reload_dataloaders_every_n_epochs'])
        #     trainer = pl.Trainer(max_epochs = cfg['epoch'], gpus= 1,logger=logger)
        #     trainer.fit(model.to(device), dm)
        # except RuntimeError as e:
        #     print(f"GPU {device_idx} is busy. Trying next GPU.")
        #     print(e)
    # trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model.to(device), dm)
