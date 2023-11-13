from Model.LitModel import LitModel
from Dataset.Datamodule import CROHMEDatamodule
from make_data import make_data
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import optuna

import argparse
import os
import sys



def objective(trial: optuna.trial.Trial):
    stroke_emb_nb = 150
    rel_emb_nb = 10
    stroke_emb_nb = trial.suggest_int('stroke_emb_nb', 50, 201, step=50)
    rel_emb_nb = trial.suggest_int('rel_emb_nb', 5, 11, step=5)
    # batch_size = trial.suggest_int('batch_size', 96, 257, step=32)
    batch_size = 256
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    lambda1 = trial.suggest_float('lambda1', 0, 1, step=0.1)
    lambda2 = 1 - lambda1
    dropout = trial.suggest_float('dropout', 0.4, 1, step=0.1)
    node_gat_input_size = 96
    edge_gat_input_size = 64
    node_gat_hidden_size = 512
    edge_gat_hidden_size = 384
    node_gat_output_size = 224
    edge_gat_output_size = 192
    gat_n_heads = 4
    # node_gat_input_size = trial.suggest_int('node_gat_input_size', 32, 257, step=32)
    # edge_gat_input_size = trial.suggest_int('edge_gat_input_size', 32, 257, step=32)
    # node_gat_hidden_size = trial.suggest_int('node_gat_hidden_size', 64, 513, step=64)
    # edge_gat_hidden_size = trial.suggest_int('edge_gat_hidden_size', 64, 513, step=64)
    # node_gat_output_size = trial.suggest_int('node_gat_output_size', 32, 257, step=32)
    # edge_gat_output_size = trial.suggest_int('edge_gat_output_size', 32, 257, step=32)
    # gat_n_heads = trial.suggest_int('gat_n_heads', 4, 9, step=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    speed = False
    epoch = 300

    if device == 'cpu':
        data_path = '/home/e19b516g/yejing/data/data_for_graph/'
    else:
        data_path = '/home/xie-y/data/Edge_GAT/'
    npz_name = 'S'+ str(stroke_emb_nb) + '_R' + str(rel_emb_nb) + '_Speed_' + str(speed)
    npz_path = os.path.join(data_path, npz_name)
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
        make_data(os.path.join(data_path, 'INKML'), npz_path, stroke_emb_nb, rel_emb_nb, speed)
    root_path = sys.path[0]
    results_path = os.path.join(root_path, 'val_results')
    exp_name = 'lr_' + str(lr) + '_bs_' + str(batch_size) + '_epoch_' + str(epoch) + '_dropout_' + str(dropout) + '_l1_' + str(lambda1) + '_l2_' + str(lambda2)
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
        node_input_size = stroke_emb_nb,
        edge_input_size = rel_emb_nb * 4,
        node_gat_input_size = 128,
        edge_gat_input_size = 256,
        node_gat_hidden_size = node_gat_hidden_size,
        edge_gat_hidden_size = edge_gat_hidden_size,
        node_gat_output_size = node_gat_output_size,
        edge_gat_output_size = edge_gat_output_size,
        gat_n_heads = gat_n_heads,
        node_class_nb = 114,
        edge_class_nb = 14,
        dropout = dropout,
        lr = lr,
        lambda1 = lambda1,
        lambda2 = lambda2,
        patience = 30,
        min_delta = 0.00001,
        results_path = val_results_version
    )
    dm = CROHMEDatamodule(
        root_path = npz_path,
        shuffle = True,
        batch_size = batch_size,
        num_workers = 8,
        max_node = -1,
        reload_dataloaders_every_n_epochs = 1
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
        callbacks=[optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_acc_node'), early_stopping]
    )
    hyperparameters = dict(stroke_emb_nb=stroke_emb_nb, rel_emb_nb=rel_emb_nb, batch_size=batch_size, lr=lr, lambda1=lambda1, lambda2=lambda2, dropout=dropout, node_gat_input_size=node_gat_input_size, edge_gat_input_size=edge_gat_input_size, node_gat_hidden_size=node_gat_hidden_size, edge_gat_hidden_size=edge_gat_hidden_size, node_gat_output_size=node_gat_output_size, edge_gat_output_size=edge_gat_output_size, gat_n_heads=gat_n_heads)
    trainer.logger.log_hyperparams(hyperparameters)
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