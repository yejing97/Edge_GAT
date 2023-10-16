import torch
import pytorch_lightning as pl

from MainModel import MainModel

class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lr = args.lr
        self.model = MainModel(args.node_input_size, args.edge_input_size, args.node_emb_size, args.gat_input_size, args.gat_hidden_size, args.gat_output_size, args.gat_n_heads, args.node_class_nb, args.edge_class_nb, args.dropout)
    
    def training_step(self, batch):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        loss_node = torch.nn.functional.cross_entropy(node_hat, strokes_label)
        loss_edge = torch.nn.functional.cross_entropy(edge_hat, edges_label)
        loss = self.lambda1*loss_node + self.lambda2 * loss_edge
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    