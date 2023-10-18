from Model.LitModel import LitModel
from Dataset.Datamodule import CROHMEDatamodule
import pytorch_lightning as pl

dm = CROHMEDatamodule(
    root_path = '/home/e19b516g/yejing/data/data_for_graph/npz/',
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
    lambda1 = 0.5,
    lambda2 = 0.5,
    lr = 0.001
)

trainer = pl.Trainer()
trainer.fit(model, dm)