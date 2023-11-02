import torch
import pytorch_lightning as pl
from tsai.all import *
from labml_nn.graphs.gat import GraphAttentionLayer
from sklearn.metrics import accuracy_score
from collections import OrderedDict

# import sys
from Model.EdgeGat import Readout
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
        self.d = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt_path = args['ckpt_path']

        self.loss = torch.nn.CrossEntropyLoss()
        self.node_emb_model = XceptionTime(self.node_input_size, self.gat_input_size)
        # if os.path.isfile(self.ckpt_path):
        #     self.node_emb_model.load_state_dict(self.load_ckpt(self.ckpt_path), strict=False)
        self.gat1 = GraphAttentionLayer(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout=self.dropout)
        self.gat2 = GraphAttentionLayer(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout=self.dropout)
        self.readout_node = Readout(self.gat_output_size, self.node_class_nb)
        # self.model = MainModel(self.node_input_size, self.edge_input_size, self.gat_input_size, self.gat_hidden_size, self.gat_output_size, self.gat_n_heads, self.node_class_nb, self.edge_class_nb, self.dropout)
    
    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.d)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            # print(k, v.shape)
            # print(k[:6] + 'node_emb.' + k[6:])
            new_state_dict[k[:6] + 'node_emb.' + k[6:]] = v
        return new_state_dict

    def training_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        strokes_label = strokes_label.squeeze(0)
        indices = torch.nonzero(los.reshape(-1)).squeeze()
        edges_label = edges_label.squeeze(0).reshape(-1)[indices]

        # add self connection for los
        los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)

        # node_hat, edge_hat = self.model(strokes_emb.to(self.d), edges_emb.to(self.d), los.to(self.d))
        node_gat_feat = self.node_emb_model(strokes_emb.to(self.d).squeeze(0))
        # node_gat_feat = self.gat1(node_gat_feat, los.to(self.d))
        # node_gat_feat = self.gat2(node_gat_feat, los.to(self.d))
        # node_hat = self.readout_node(node_gat_feat)
                
        # edge_hat = edge_hat.reshape(-1, self.edge_class_nb)[indices]
        # print(edge_hat)
        # print(edges_label)
        loss = self.loss(node_gat_feat, strokes_label)
        # loss_edge = self.loss(edge_hat, edges_label)
        # if loss_edge.isnan():
        #     loss = loss_node
        # else:
        #     # loss = self.lambda1*loss_node + self.lambda2 * loss_edge
        #     loss = loss_node
        if loss.isnan():
            print('node_hat', node_gat_feat)
            # print('edge_hat', edge_hat)
            print(los)
        # self.log('train_loss_node', loss_node)
        # self.log('train_loss_edge', loss_edge)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        strokes_label = strokes_label.squeeze(0)
        indices = torch.nonzero(los.reshape(-1)).squeeze()
        edges_label = edges_label.squeeze(0).reshape(-1)[indices]

        # add self connection for los
        los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)


        # node_hat, edge_hat = self.model(strokes_emb.to(self.d), edges_emb.to(self.d), los.to(self.d))
        # edge_hat = edge_hat.reshape(-1, self.edge_class_nb)[indices]
        node_gat_feat = self.node_emb_model(strokes_emb.to(self.d).squeeze(0))
        # node_gat_feat = self.gat1(node_gat_feat, los.to(self.d))
        # node_gat_feat = self.gat2(node_gat_feat, los.to(self.d))
        # node_hat = self.readout_node(node_gat_feat)
        
        loss = self.loss(node_gat_feat, strokes_label)
        # loss_node = self.loss(node_hat, strokes_label)
        # loss_edge = self.loss(edge_hat, edges_label)
        # print(strokes_label)
        # loss = self.lambda1*loss_node + self.lambda2 * loss_edge
        acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_gat_feat, dim=1).cpu().numpy())
        # acc_edge = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
        # self.log('val_loss_node', loss_node)
        # self.log('val_loss_edge', loss_edge)
        self.log('val_loss', loss)
        self.log('val_acc_node', acc_node)
        # self.log('val_acc_edge', acc_edge)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    