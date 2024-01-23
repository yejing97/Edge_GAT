from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import yaml
import pytorch_lightning as pl
from tsai.all import *
# from labml_nn.graphs.gat import GraphAttentionLayer
from sklearn.metrics import accuracy_score, precision_score
from collections import OrderedDict

# import sys
from Model.EdgeGat import Readout
from Model.MainModel import MainModel
from Model.focalloss import FocalLoss

torch.set_printoptions(threshold=np.inf)

class LitModel(pl.LightningModule):
    def __init__(self, config_path, mode, results_path):
        super().__init__()
        self.mode = mode
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.lr = float(config['lr'])
        self.node_input_size = config['stroke_emb_nb']
        self.edge_input_size = config['rel_emb_nb']
        # self.node_gat_input_size = config['node_gat_input_size']
        # self.edge_gat_input_size = config['edge_gat_input_size']
        # self.node_gat_hidden_size = config['node_gat_hidden_size']
        # self.edge_gat_hidden_size = config['edge_gat_hidden_size']
        # self.node_gat_output_size = config['node_gat_output_size']
        # self.edge_gat_output_size = config['edge_gat_output_size']
        self.gat_heads_parm = config['gat_heads_parm']
        self.node_class_nb = config['node_class_nb']
        self.edge_class_nb = config['edge_class_nb']
        self.dropout = config['dropout']
        self.patience = config['patience']
        self.min_delta = float(config['min_delta'])
        self.loss_gamma = config['loss_gamma']
        self.d = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_path = results_path

        self.loss_node = torch.nn.CrossEntropyLoss()
        # self.loss_edge = torch.nn.CrossEntropyLoss()
        # self.loss_node = FocalLoss(gamma=2)
        self.loss_edge = FocalLoss(gamma=2)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.node_correct = 0
        self.edge_correct = 0
        self.all_correct = 0
        # self.node_emb_model = XceptionTime(self.node_input_size, self.gat_input_size)
        # if os.path.isfile(self.ckpt_path):
        #     self.node_emb_model.load_state_dict(self.load_ckpt(self.ckpt_path), strict=False)
        # self.gat1 = GraphAttentionLayer(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout=self.dropout)
        # self.gat2 = GraphAttentionLayer(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout=self.dropout)
        # self.readout_node = Readout(self.gat_output_size, self.node_class_nb)

        # self.model = MainModel(self.node_input_size, self.edge_input_size, self.gat_input_size, self.gat_hidden_size, self.gat_output_size, self.gat_n_heads, self.node_class_nb, self.edge_class_nb, self.dropout)
        self.model = MainModel(
            node_input_size = config['stroke_emb_nb'],
            edge_input_size = config['rel_emb_nb'],
            edge_emb_parm=config['edge_emb_parm'],
            node_gat_parm=config['node_gat_parm'],
            edge_gat_parm=config['edge_gat_parm'],
            node_class_nb = config['node_class_nb'],
            edge_class_nb = config['edge_class_nb'],
            gat_heads_parm = config['gat_heads_parm'],
            node_readout=config['node_readout'],
            edge_readout=config['edge_readout'],
            dropout = config['dropout'],
            mode=mode
            )
    
    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.d)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            # print(k, v.shape)
            # print(k[:6] + 'node_emb.' + k[6:])
            new_state_dict[k[:6] + 'node_emb.' + k[6:]] = v
        return new_state_dict

    def load_batch(self, batch):
        name, strokes_emb, edges_emb, los, strokes_label, edges_label, mask = batch
        # strokes_emb = strokes_emb.squeeze(0)
        strokes_emb = strokes_emb.reshape(strokes_emb.shape[0]*strokes_emb.shape[1], strokes_emb.shape[2], strokes_emb.shape[3])
        edges_emb = edges_emb.reshape(edges_emb.shape[0],edges_emb.shape[1], edges_emb.shape[2], -1)
        strokes_label = strokes_label.long().reshape(-1)
        edges_label = edges_label.long()
        mask = mask.squeeze(0).reshape(-1)
        # los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)
        los = los.to(self.d) + torch.eye(los.shape[1], los.shape[2]).repeat(los.shape[0], 1, 1).to(self.d)
        new_los = torch.zeros((los.shape[0]*los.shape[1], los.shape[0]*los.shape[2]))
        new_edges_label = torch.zeros((edges_label.shape[0]*edges_label.shape[1], edges_label.shape[0]*edges_label.shape[2])).long()
        new_edges_emb = torch.zeros((edges_emb.shape[0]*edges_emb.shape[1], edges_emb.shape[0]*edges_emb.shape[2], edges_emb.shape[3]))
        for i in range(los.shape[0]):
            new_los[i*los.shape[1]:(i+1)*los.shape[1], i*los.shape[1]:(i+1)*los.shape[1]] = los[i]
            new_edges_label[i*edges_label.shape[1]:(i+1)*edges_label.shape[1], i*edges_label.shape[1]:(i+1)*edges_label.shape[1]] = edges_label[i]
            new_edges_emb[i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1], i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1]] = edges_emb[i]
        return strokes_emb.to(self.d), new_edges_emb.to(self.d), new_los.unsqueeze(-1).to(self.d), strokes_label.to(self.d), new_edges_label.to(self.d).reshape(-1), mask.to(self.d), name
    
    def edge_filter(self, edges_emb, edges_label, los):
        if edges_emb.shape[1] == 2:
            # print('edge class = 2')
            edges_label = torch.where(edges_label == 1, 1, 0)
        elif edges_emb.shape[1] == 14:
            edges_label = torch.where(edges_label < 14, edges_label, 0)
        try:
            los = los.squeeze().fill_diagonal_(0)
            los = torch.triu(los)
        except:
            los = torch.zeros(0)
        indices = torch.nonzero(los.reshape(-1)).squeeze().reshape(-1)
        edges_label = edges_label[indices]
        edges_emb = edges_emb.reshape(-1, edges_emb.shape[-1])[indices]
        return edges_emb, edges_label
    
    def node_filter(self, node_emb, node_label):
        indices = torch.nonzero(node_label != 0).squeeze()
        node_label = node_label[indices]
        node_emb = node_emb[indices]
        return node_emb, node_label
    
    def node_mask(self, node_emb, node_label, mask):
        indices = torch.nonzero(mask == 1).squeeze()
        node_label = node_label[indices]
        node_emb = node_emb[indices]
        return node_emb, node_label
    
    def edge_mask(self, edges_emb, edges_label, los):
        if edges_emb.shape[1] == 2:
            # print('edge class = 2')
            edges_label = torch.where(edges_label == 1, 1, 0)
        elif edges_emb.shape[1] == 14:
            edges_label = torch.where(edges_label < 14, edges_label, 0)
        try:
            los = los.squeeze().fill_diagonal_(0)
            los = torch.triu(los)
        except:
            los = torch.zeros(0)
        edges_label = edges_label.reshape(los.shape[0], los.shape[1])
        edges_emb = edges_emb.reshape(los.shape[0], los.shape[1], -1)
        edges_label = edges_label * los
        edges_emb = edges_emb * los.unsqueeze(-1)
        return edges_emb, edges_label
    
    def am_for_pretrain(self, edges_label, los):
        am = torch.where(edges_label == 1, 1, 0).reshape(los.shape[0], los.shape[1], los.shape[2])
        for i in range(los.shape[0]):
            am[i][i] = 1
        return am
    
    
    def training_step(self, batch, batch_idx):
        try:
            strokes_emb, edges_emb, los, strokes_label, edges_label, mask, _ = self.load_batch(batch)
            # if self.mode == 'pre_train':
            #     # los = self.am_for_pretrain(edges_label, los)
            #     # return
            #     node_hat= self.model(strokes_emb, edges_emb, los)
            #     node_hat, strokes_label = self.node_mask(node_hat, strokes_label, mask)
            #     loss_node = self.loss_node(node_hat, strokes_label)
            #     self.log('train_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
            #     return loss_node
            # else:
            node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
            # try:
                # print(torch.where(edges_label == 0, 1, 0).sum())
            node_hat, strokes_label = self.node_mask(node_hat, strokes_label, mask)
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)

            loss_edge = self.loss_edge(edge_hat, edges_label)

            loss_node = self.loss_node(node_hat, strokes_label)
            # loss_edge = self.loss_edge(edge_hat, edges_label)
            loss = self.lambda1*loss_node + self.lambda2 * loss_edge

            

            if self.mode == 'pre_train':
                self.log('train_loss', loss_edge, on_epoch=True, prog_bar=True, logger=True)
            else:
                self.log('train_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
                self.log('train_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
                self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            return loss
        except:
            print('error with batch ' + str(batch_idx))
            return
    
    def validation_step(self, batch, batch_idx):
        try:
            strokes_emb, edges_emb, los, strokes_label, edges_label, mask, _ = self.load_batch(batch)

        except:
            print('error with batch ' + str(batch_idx))
            print('stroke_emb', batch[0].shape)
            return
        # if self.mode == 'pre_train':
        #     # los = self.am_for_pretrain(edges_label, los)
        #     node_hat= self.model(strokes_emb, edges_emb, los)
        #     node_hat, strokes_label = self.node_mask(node_hat, strokes_label, mask)
        #     loss_node = self.loss_node(node_hat, strokes_label)
        #     self.log('val_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
        #     acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
        #     self.log('val_acc_node', acc_node, on_epoch=True, prog_bar=True, logger=True)
        #     return acc_node
        # else:
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        try:
            node_hat, strokes_label = self.node_mask(node_hat, strokes_label, mask)
            edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
            # print(torch.where(edges_label == 0, 1, 0).sum())

            loss_edge = self.loss_edge(edge_hat, edges_label)
            loss_node = self.loss_node(node_hat, strokes_label)
            loss = self.lambda1*loss_node + self.lambda2 * loss_edge
            self.validation_step_outputs.append([node_hat, strokes_label, edge_hat, edges_label])

            acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
            acc_edge = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
            self.log("val_loss_node", loss_node, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss_edge', loss_edge, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('val_acc_node', acc_node, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_acc_edge', acc_edge, on_epoch=True, prog_bar=True, logger=True)

            return acc_node
        except:
            print('error with batch ' + str(batch_idx))
            return
    def test_step(self, batch, batch_idx):
        # try:
        strokes_emb, edges_emb, los, strokes_label, edges_label, mask, name = self.load_batch(batch)
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        try:
            edge_hat_save, edges_label_save = self.edge_mask(edge_hat, edges_label, los)
            torch.save([node_hat, edge_hat_save,strokes_label, edges_label_save, los], os.path.join(self.results_path, 'test',name[0].split('.')[0] + '.pt'))
        except:
            print('error with ' + str(name))
            return
        
        edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
        
        # edges_label = edges_label.reshape(-1)
        # edge_hat = edge_hat.reshape(-1, edge_hat.shape[-1])
        acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
        acc_edge = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
        prec_node = precision_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy(),average='weighted')
        prec_edge = precision_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy(),average='weighted')

        self.log('test_acc_node', acc_node, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_edge', acc_edge, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_prec_node', prec_node, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_prec_edge', prec_edge, on_epoch=True, prog_bar=True, logger=True)
        if acc_node > 0.98:
            self.node_correct += 1
        if acc_edge == 1:
            self.edge_correct += 1
        if acc_node == 1 and acc_edge == 1:
            self.all_correct += 1
        return acc_node
    
    def on_test_epoch_end(self) -> None:
        print('node acc: ' + str(self.node_correct))
        print('edge acc: ' + str(self.edge_correct))
        print('all acc: ' + str(self.all_correct))
        

    
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
        if self.mode == 'pre_train_node':
            return optimizer
        elif self.mode == 'pre_train_edge':
            return optimizer
        elif self.mode == 'train':
            return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True, mode='max'),
            'monitor': 'val_acc_node',
            'frequency': self.trainer.check_val_every_n_epoch
        }
    