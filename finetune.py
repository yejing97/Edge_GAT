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
parser.add_argument('--gpu_id', type=int, default = 1)
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
    node_class_nb = 102
    edge_class_nb = args.edge_class
    stroke_emb_nb = trial.suggest_categorical('stroke_emb_nb', [150])
    rel_emb_nb = trial.suggest_categorical('rel_emb_nb', [40])
    total_batch_size = trial.suggest_categorical('total_batch_size', [128, 256, 512])
    max_node = trial.suggest_categorical('max_node', [4,8,10,12,16])
    batch_size = total_batch_size // max_node
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    lambda1 = trial.suggest_float('lambda1', 0.4, 0.8, step=0.1)
    lambda2 = 1 - lambda1
    gat_layer = trial.suggest_categorical('gat_layer', [2, 3, 4, 5])
    dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)
    edge_emb_layer = trial.suggest_categorical('edge_emb_layer', [2, 3])
    readout_layer = trial.suggest_categorical('readout_layer', [2, 3, 4])
    node_gat_parm = []
    edge_gat_parm = []
    gat_heads_parm = []
    for i in range(gat_layer - 1):
        random_node = trial.suggest_categorical('random_node_' + str(i), [64, 128, 256, 384, 512])
        node_gat_parm.append(random_node)
        random_edge = trial.suggest_categorical('random_edge_' + str(i), [64, 128, 256, 384, 512])
        edge_gat_parm.append(random_edge)
        random_heads = trial.suggest_categorical('random_heads_' + str(i), [4, 8, 16])
        gat_heads_parm.append(random_heads)
    node_gat_parm.append(trial.suggest_categorical('node_gat_parm_' + str(gat_layer - 1), [128, 256, 384, 512]))
    edge_gat_parm.append(trial.suggest_categorical('edge_gat_parm_' + str(gat_layer - 1), [128, 256, 384, 512]))
    gat_heads_parm.append(1)

    edge_emb_parm = []
    edge_emb_parm.append(rel_emb_nb)
    for i in range(edge_emb_layer - 2):
        random_edge = trial.suggest_categorical('random_edge_' + str(i), [64, 128, 256, 384, 512])
        edge_emb_parm.append(random_edge)
    edge_emb_parm.append(edge_gat_parm[0])

    node_readout = []
    edge_readout = []
    node_readout.append(node_gat_parm[-1])
    edge_readout.append(edge_gat_parm[-1] * 2)
    for i in range(readout_layer - 1):
        random_node = trial.suggest_categorical('random_node_' + str(i), [64, 128, 256, 384, 512])
        node_readout.append(random_node)
        random_edge = trial.suggest_categorical('random_edge_' + str(i), [64, 128, 256, 384, 512])
        edge_readout.append(random_edge)
    node_readout.append(node_class_nb)
    edge_readout.append(edge_class_nb)

    dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)

    reload_dataloaders_every_n_epochs = args.reload_dataloaders_every_n_epochs
    loss_gamma = trial.suggest_float('loss_gamma', 1, 3, step=0.5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    speed = False
    epoch = 50

    hyperparameters = dict(
        stroke_emb_nb=stroke_emb_nb,
        rel_emb_nb=rel_emb_nb,
        batch_size=batch_size,
        max_node=max_node,
        lr=lr,
        lambda1=lambda1,
        lambda2=lambda2,
        dropout=dropout,
        edge_gat_parm=edge_gat_parm,
        node_gat_parm=node_gat_parm,
        gat_heads_parm=gat_heads_parm,
        edge_emb_parm=edge_emb_parm,
        node_readout=node_readout,
        edge_readout=edge_readout,
        loss_gamma=loss_gamma,
        patience=5,
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
        logger_path = os.path.join(root_path,'finetunning/finetunning_best', hyp_name, npz_name)
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
        torch.cuda.set_device(args.gpu_id)

        trainer = pl.Trainer(
            max_epochs=epoch,
            logger=logger,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            callbacks=[optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_acc_node'), early_stopping]
        )
        try:
            trainer.fit(model.to(device), dm)
            return trainer.callback_metrics['val_acc_node'].item()

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