from Model.LitModel import LitModel
from Model.MOModel import MOModel
from Dataset.Datamodule import CROHMEDatamodule
from make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
import yaml
import time

import argparse
import os
import sys

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
    rel_emb_nb = trial.suggest_categorical('rel_emb_nb', [5, 10])
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    max_node = trial.suggest_int('max_node', 2,10, step=2)
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    lambda1 = trial.suggest_float('lambda1', 0, 1, step=0.1)
    lambda2 = 1 - lambda1
    dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.1)
    gat_n_heads = trial.suggest_categorical('gat_n_heads', [1, 2, 4, 8])
    node_gat_input_size = trial.suggest_categorical('node_gat_input_size', [32, 64, 128, 256])
    edge_gat_input_size = trial.suggest_categorical('edge_gat_input_size', [32, 64, 128, 256])
    node_gat_hidden_size = trial.suggest_categorical('node_gat_hidden_size', [32, 64, 128, 256])
    edge_gat_hidden_size = trial.suggest_categorical('edge_gat_hidden_size', [32, 64, 128, 256])
    node_gat_output_size = trial.suggest_categorical('node_gat_output_size', [32, 64, 128, 256])
    edge_gat_output_size = trial.suggest_categorical('edge_gat_output_size', [32, 64, 128, 256])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    speed = False
    epoch = 300

    hyperparameters = dict(
        stroke_emb_nb=stroke_emb_nb,
        rel_emb_nb=rel_emb_nb,
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
        patience=30,
        min_delta=0.00001,
        shuffle=True,
        num_workers=0,
        node_class_nb=114,
        edge_class_nb=26,
        epoch = 300
        )



    if device == 'cpu':
        data_path = '/home/e19b516g/yejing/data/data_for_graph/'
    else:
        data_path = '/home/xie-y/data/Edge_GAT/'
    npz_name = 'S'+ str(stroke_emb_nb) + '_R' + str(rel_emb_nb)
    npz_path = os.path.join(data_path, npz_name)
    if not os.path.exists(npz_path):
        print('npz_path not exists')
        return 0.0
        # os.makedirs(npz_path)
        # make_data(os.path.join(data_path, 'INKML'), npz_path, stroke_emb_nb, rel_emb_nb, speed, 'stroke')
    root_path = sys.path[0]
    results_path = os.path.join(root_path, 'val_results')
    # use time to give a unique name for each experiment
    exp_name = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    yaml_path = os.path.join(root_path, 'config', npz_name + '_' + exp_name + '.yaml')
    make_yaml(hyperparameters, yaml_path)
    # exp_name = 'lr_' + str(lr) + '_bs_' + str(batch_size) + '_epoch_' + str(epoch) + '_dropout_' + str(dropout) + '_l1_' + str(lambda1) + '_l2_' + str(lambda2)
    logger_path = os.path.join(root_path, 'finetunning' , npz_name)
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
        monitor='val_acc_node',
        min_delta=0,
        patience=40,
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=epoch,
        accelerator="auto",
        devices=1,
        logger=logger,
        reload_dataloaders_every_n_epochs=10,
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