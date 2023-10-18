import torch
import pytorch_lightning as pl
import sys
print(sys.path)

from Model.MainModel import MainModel
class LitModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lr = args['lr']
        self.node_input_size = args['node_input_size']
        self.edge_input_size = args['edge_input_size']
        self.gat_input_size = args['gat_input_size']
        self.gat_hidden_size = args['gat_hidden_size']
        self.gat_output_size = args['gat_output_size']
        self.gat_n_heads = args['gat_n_heads']
        self.node_class_nb = args['node_class_nb']
        self.edge_class_nb = args['edge_class_nb']
        self.dropout = args['dropout']
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = MainModel(self.node_input_size, self.edge_input_size, self.gat_input_size, self.gat_hidden_size, self.gat_output_size, self.gat_n_heads, self.node_class_nb, self.edge_class_nb, self.dropout)
    
    def training_step(self, batch):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        # print('stroke_emb', strokes_emb.squeeze(0).shape)
        # print('strokes_label', strokes_label.squeeze(0).shape)
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        # print('node_hat', node_hat.shape)
        loss_node = self.loss(node_hat, strokes_label.squeeze(0))
        loss_edge = self.loss(edge_hat, edges_label.squeeze(0).reshape(-1))
        loss = self.lambda1*loss_node + self.lambda2 * loss_edge
        self.log('train_loss_node', loss_node)
        self.log('train_loss_edge', loss_edge)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        loss_node = self.loss(node_hat, strokes_label.squeeze(0))
        loss_edge = self.loss(edge_hat, edges_label.squeeze(0).reshape(-1))
        loss = self.lambda1*loss_node + self.lambda2 * loss_edge
        self.log('val_loss_node', loss_node)
        self.log('val_loss_edge', loss_edge)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    