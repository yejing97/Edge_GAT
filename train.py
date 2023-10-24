from Model.LitModel import LitModel
from Dataset.Datamodule import CROHMEDatamodule
import pytorch_lightning as pl

dm = CROHMEDatamodule(
    root_path = '/home/xie-y/data/npz/',
    shuffle = True,
    num_workers = 8
)

model = LitModel(
    node_input_size = 100,
    edge_input_size = 20,
    gat_input_size = 128,
    gat_hidden_size = 64,
    gat_output_size = 128,
    gat_n_heads = 8,
    node_class_nb = 114,
    edge_class_nb = 14,
    dropout = 0.6,
    lambda1 = 0.1,
    lambda2 = 0.9,
    lr = 0.001
)

trainer = pl.Trainer(accelerator="gpu", max_epochs=100, gpus=1, progress_bar_refresh_rate=1)
trainer.fit(model, dm)