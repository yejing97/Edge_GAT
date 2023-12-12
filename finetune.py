from Model.LitModel import LitModel
from Model.MOModel import MOModel
from Dataset.Datamodule import CROHMEDatamodule
# from make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
import yaml
import time

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--edge_class', type=int, default=14)
parser.add_argument('--am_type', type=str, default='los')
parser.add_argument('--node_norm', type=bool, default=True)
parser.add_argument('--edge_feat', type=str, default='R10')
parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default = 5)
args = parser.parse_args()
def make_yaml(hyperparameters, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(hyperparameters, file, default_flow_style=False)
    print('yaml file saved in ', yaml_path)
    print('hyperparameters: ', hyperparameters)

def objective(trial: optuna.trial.Trial):
    # stroke_emb_nb = 150
    # rel_emb_nb = 10
    # stroke_emb_nb = trial.suggest_int('stroke_emb_nb', 100, 151, step=50)
    stroke_emb_nb = trial.suggest_categorical('stroke_emb_nb', [150])
    # rel_emb_nb = trial.suggest_int('rel_emb_nb', 5, 11, step=5)
    rel_emb_nb = trial.suggest_categorical('rel_emb_nb', [10])
    total_batch_size = trial.suggest_categorical('total_batch_size', [128, 256])
    # batch_size = trial.suggest_categorical('batch_size', [16])
    max_node = trial.suggest_categorical('max_node', [4,8,16])
    # max_node = trial.suggest_categorical('max_node', 16)
    batch_size = total_batch_size // max_node
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    if args.edge_class == 2:

        lambda1 = trial.suggest_float('lambda1', 0.1, 0.3, step=0.05)
    # lambda1 = 0.6
    # lambda2 = 1 - lambda1
        lambda2 = trial.suggest_float('lambda2', 1, 10, step=1)
    else:
        lambda1 = trial.suggest_float('lambda1', 0.4, 0.8, step=0.1)
        lambda2 = 1 - lambda1
    # dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)
    dropout = trial.suggest_categorical('dropout', [0.3])
    # gat_n_heads = trial.suggest_categorical('gat_n_heads', [4, 8])
    gat_n_heads = 8
    node_gat_input_size = trial.suggest_categorical('node_gat_input_size', [64, 128, 256])
    edge_gat_input_size = trial.suggest_categorical('edge_gat_input_size', [64, 128, 256])
    node_gat_hidden_size = trial.suggest_categorical('node_gat_hidden_size', [64, 128, 256])
    edge_gat_hidden_size = trial.suggest_categorical('edge_gat_hidden_size', [64, 128, 256])
    node_gat_output_size = trial.suggest_categorical('node_gat_output_size', [64, 128, 256])
    edge_gat_output_size = trial.suggest_categorical('edge_gat_output_size', [64, 128, 256])

    reload_dataloaders_every_n_epochs = args.reload_dataloaders_every_n_epochs

    # node_gat_input_size = 128
    # edge_gat_input_size = 64
    # node_gat_hidden_size = 256
    # edge_gat_hidden_size = 64
    # node_gat_output_size = 128
    # edge_gat_output_size = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    speed = False
    epoch = 100

    hyperparameters = dict(
        stroke_emb_nb=stroke_emb_nb,
        rel_emb_nb=rel_emb_nb * 4,
        batch_size=batch_size,
        max_node=max_node,
        lr=lr,
        lambda1=lambda1,
        lambda2=lambda2,
        dropout=dropout,
        gat_n_heads=gat_n_heads,
        node_gat_input_size=node_gat_input_size,
        edge_gat_input_size=edge_gat_input_size,
        node_gat_hidden_size=node_gat_hidden_size,
        edge_gat_hidden_size=edge_gat_hidden_size,
        node_gat_output_size=node_gat_output_size,
        edge_gat_output_size=edge_gat_output_size,
        loss_gamma=2,
        patience=10,
        min_delta=1e-4,
        shuffle=True,
        num_workers=0,
        node_class_nb=102,
        edge_class_nb=args.edge_class,
        epoch = epoch,
        am_type = args.am_type,
        node_norm = args.node_norm,
        reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
        )



    if device == 'cpu':
        data_path = '/home/e19b516g/yejing/data/data_for_graph/'
    else:
        data_path = '/home/xie-y/data/Edge_GAT/'
    npz_name = 'S'+ str(stroke_emb_nb) + '_' + args.edge_feat
    npz_path = os.path.join(data_path, npz_name)
    if not os.path.exists(npz_path):
        print('npz_path not exists')
        return 0.0
        # os.makedirs(npz_path)
        # make_data(os.path.join(data_path, 'INKML'), npz_path, stroke_emb_nb, rel_emb_nb, speed, 'stroke')
    else:
        root_path = sys.path[0]
        results_path = os.path.join(root_path, 'val_results')
        # use time to give a unique name for each experiment
        exp_name = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        yaml_path = os.path.join(root_path, 'config', npz_name + '_' + exp_name + '.yaml')
        make_yaml(hyperparameters, yaml_path)
        # exp_name = 'lr_' + str(lr) + '_bs_' + str(batch_size) + '_epoch_' + str(epoch) + '_dropout_' + str(dropout) + '_l1_' + str(lambda1) + '_l2_' + str(lambda2)
        hyp_name = args.am_type + '_nodenorm_' + str(args.node_norm) + '_edgeclass_' + str(args.edge_class)
        logger_path = os.path.join(root_path,'finetunning_14_nopadding', hyp_name, npz_name)
        logger = TensorBoardLogger(save_dir=logger_path, name=exp_name)
        val_results_path = os.path.join(results_path, npz_name, exp_name)
        if not os.path.exists(val_results_path):
            os.makedirs(val_results_path)
            version = 0
        else:
            version = len(os.listdir(val_results_path))
        val_results_version = os.path.join(val_results_path, 'version_' + str(version))
        os.makedirs(val_results_version)

        model = LitModel(
            config_path = yaml_path,
            mode = 'train',
            results_path = val_results_version
        )
        dm = CROHMEDatamodule(
            npz_path = npz_path,
            config_path = yaml_path
        )
        early_stopping = pl.callbacks.EarlyStopping(
            monitor='val_acc_edge',
            min_delta=0,
            patience=20,
            verbose=False,
            mode='max'
        )

        trainer = pl.Trainer(
            max_epochs=epoch,
            accelerator="auto",
            devices=1,
            logger=logger,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            callbacks=[optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_acc_edge'), early_stopping]
        )
        try:
            trainer.fit(model.to(device), dm)
            return trainer.callback_metrics['val_acc_edge'].item()

        except Exception as e:
            print(f"An exception occurred during training: {str(e)}")
            return 0.0



if __name__ == "__main__":


    pruner = optuna.pruners.MedianPruner() 

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))