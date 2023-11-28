import torch
import pytorch_lightning as pl
from tsai.all import *
# from labml_nn.graphs.gat import GraphAttentionLayer
from sklearn.metrics import accuracy_score
from collections import OrderedDict

# import sys
from Model.EdgeGat import Readout
from Model.MainModel import MainModel
from Model.focalloss import FocalLoss

torch.set_printoptions(threshold=np.inf)

class LitModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.mode = args['mode']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lr = args['lr']
        self.node_input_size = args['node_input_size']
        self.edge_input_size = args['edge_input_size']
        # self.gat_input_size = args['gat_input_size']
        self.node_gat_input_size = args['node_gat_input_size']
        self.edge_gat_input_size = args['edge_gat_input_size']
        # self.gat_hidden_size = args['gat_hidden_size']
        self.node_gat_hidden_size = args['node_gat_hidden_size']
        self.edge_gat_hidden_size = args['edge_gat_hidden_size']
        # self.gat_output_size = args['gat_output_size']
        self.node_gat_output_size = args['node_gat_output_size']
        self.edge_gat_output_size = args['edge_gat_output_size']
        self.gat_n_heads = args['gat_n_heads']
        self.node_class_nb = args['node_class_nb']
        self.edge_class_nb = args['edge_class_nb']
        self.dropout = args['dropout']
        self.patience = args['patience']
        self.min_delta = args['min_delta']
        self.d = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.ckpt_path = args['ckpt_path']
        self.results_path = args['results_path']

        self.loss_node = torch.nn.CrossEntropyLoss()
        self.loss_edge = FocalLoss(gamma=2)

        self.validation_step_outputs = []
        # self.node_emb_model = XceptionTime(self.node_input_size, self.gat_input_size)
        # if os.path.isfile(self.ckpt_path):
        #     self.node_emb_model.load_state_dict(self.load_ckpt(self.ckpt_path), strict=False)
        # self.gat1 = GraphAttentionLayer(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout=self.dropout)
        # self.gat2 = GraphAttentionLayer(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout=self.dropout)
        # self.readout_node = Readout(self.gat_output_size, self.node_class_nb)

        # self.model = MainModel(self.node_input_size, self.edge_input_size, self.gat_input_size, self.gat_hidden_size, self.gat_output_size, self.gat_n_heads, self.node_class_nb, self.edge_class_nb, self.dropout)
        self.model = MainModel(self.node_input_size, self.edge_input_size, self.node_gat_input_size, self.edge_gat_input_size, self.node_gat_hidden_size, self.edge_gat_hidden_size, self.node_gat_output_size, self.edge_gat_output_size, self.gat_n_heads, self.node_class_nb, self.edge_class_nb, self.dropout, self.mode)
    
    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.d)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            # print(k, v.shape)
            # print(k[:6] + 'node_emb.' + k[6:])
            new_state_dict[k[:6] + 'node_emb.' + k[6:]] = v
        return new_state_dict

    def load_batch(self, batch):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        strokes_emb = strokes_emb.squeeze(0)
        strokes_emb = strokes_emb.reshape(strokes_emb.shape[0]*strokes_emb.shape[1], strokes_emb.shape[2], strokes_emb.shape[3])
        edges_emb = edges_emb.squeeze(0).reshape(edges_emb.shape[0],edges_emb.shape[1], edges_emb.shape[2], edges_emb.shape[3]* edges_emb.shape[4])
        strokes_label = strokes_label.squeeze(0).long().reshape(-1)
        edges_label = edges_label.squeeze(0).long()
        # los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)
        los = los + torch.eye(los.shape[1], los.shape[2]).repeat(los.shape[0], 1, 1)
        new_los = torch.zeros((los.shape[0]*los.shape[1], los.shape[0]*los.shape[2]))
        new_edges_label = torch.zeros((edges_label.shape[0]*edges_label.shape[1], edges_label.shape[0]*edges_label.shape[2])).long()
        new_edges_emb = torch.zeros((edges_emb.shape[0]*edges_emb.shape[1], edges_emb.shape[0]*edges_emb.shape[2], edges_emb.shape[3]))
        for i in range(los.shape[0]):
            new_los[i*los.shape[1]:(i+1)*los.shape[1], i*los.shape[1]:(i+1)*los.shape[1]] = los[i]
            new_edges_label[i*edges_label.shape[1]:(i+1)*edges_label.shape[1], i*edges_label.shape[1]:(i+1)*edges_label.shape[1]] = edges_label[i]
            new_edges_emb[i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1], i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1]] = edges_emb[i]
        return strokes_emb.to(self.d), new_edges_emb.to(self.d), new_los.unsqueeze(-1).to(self.d), strokes_label.to(self.d), new_edges_label.to(self.d).reshape(-1)
    
    def edge_filter(self, edges_emb, edges_label, los):
        los = los.squeeze().fill_diagonal_(0)
        los = torch.triu(los)
        indices = torch.nonzero(los.reshape(-1)).squeeze()
        edges_label = edges_label[indices]
        edges_emb = edges_emb.reshape(-1, edges_emb.shape[-1])[indices]
        return edges_emb, edges_label
    
    def training_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = self.load_batch(batch)
        if self.mode == 'pre_train_node':
            node_hat = self.model(strokes_emb, edges_emb, los)
            loss_node = self.loss_node(node_hat, strokes_label)
            loss = loss_node
            self.log('train_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
            return loss
        elif self.mode == 'pre_train_edge':
            edge_hat = self.model(strokes_emb, edges_emb, los)
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
            loss_edge = self.loss_edge(edge_hat, edges_label)
            loss = loss_edge
            self.log('train_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
            return loss_edge
        elif self.mode == 'train':
            node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
            # print(torch.where(edges_label == 0, 1, 0).sum())
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
            # print(torch.where(edges_label == 0, 1, 0).sum())
            loss_edge = self.loss_edge(edge_hat, edges_label)

            loss_node = self.loss_node(node_hat, strokes_label)
            # loss_edge = self.loss_edge(edge_hat, edges_label)
            loss = self.lambda1*loss_node + self.lambda2 * loss_edge

            self.log('train_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)

            return loss
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = self.load_batch(batch)
        if self.mode == 'pre_train_node':
            node_hat = self.model(strokes_emb, edges_emb, los)
            loss_node = self.loss_node(node_hat, strokes_label)
            acc = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
            self.log('val_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
            self.log('acc', acc, on_epoch=True, prog_bar=False, logger=True)
            return acc
        elif self.mode == 'pre_train_edge':
            edge_hat = self.model(strokes_emb, edges_emb, los)
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
            loss_edge = self.loss_edge(edge_hat, edges_label)
            acc = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
            self.log('val_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
            self.log('acc', acc, on_epoch=True, prog_bar=False, logger=True)
            return acc
        elif self.mode == 'train':
            node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
            # print(torch.where(edges_label == 0, 1, 0).sum())

            loss_edge = self.loss_edge(edge_hat, edges_label)
            loss_node = self.loss_node(node_hat, strokes_label)
            loss = self.lambda1*loss_node + self.lambda2 * loss_edge
            self.validation_step_outputs.append([node_hat, strokes_label, edge_hat, edges_label])

            acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
            acc_edge = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
            self.log("val_loss_node", loss_node, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc_node', acc_node, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_acc_edge', acc_edge, on_epoch=True, prog_bar=False, logger=True)
            return acc_node
    
    def on_validation_epoch_end(self) -> None:
        # return super().on_validation_epoch_end()
        all_preds = self.validation_step_outputs
        epoch_id = self.current_epoch
        if epoch_id % 10 == 0:
            torch.save(all_preds, os.path.join(self.results_path, 'epoch_' + str(epoch_id) + '.pt'))
        self.validation_step_outputs.clear()
        # node_preds, node_labels, edge_preds, edge_labels = all_preds
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, min_lr=self.min_delta, verbose=True),
            'monitor': 'acc',
            'frequency': self.trainer.check_val_every_n_epoch
        }
    