from typing import Any
import torch
# import tsai
from tsai.all import *
# from labml_helpers.module import Module
import pytorch_lightning as pl
from Model.EdgeGat import EdgeGraphAttention, Readout
from labml_nn.graphs.gat import GraphAttentionLayer

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Edge_emb(pl.LightningModule):
    def __init__(self,
                 edge_emb_parm: list,
                 dropout: float):
        super().__init__()
        self.edge_emb_parm = edge_emb_parm
        for i in range(len(edge_emb_parm) - 1):
            setattr(self, 'linear' + str(i), torch.nn.Linear(edge_emb_parm[i], edge_emb_parm[i+1]))
            setattr(self, 'bn' + str(i), torch.nn.BatchNorm1d(edge_emb_parm[i+1]))
            setattr(self, 'activation' + str(i), torch.nn.LeakyReLU())
            setattr(self, 'dropout' + str(i), torch.nn.Dropout(dropout))
    
    def forward(self, edge_in_features):
        for i in range(len(self.edge_emb_parm) - 1):
            edge_in_features = getattr(self, 'linear' + str(i))(edge_in_features).transpose(1, 2)
            try:
                edge_in_features = getattr(self, 'bn' + str(i))(edge_in_features).transpose(1, 2)
            except:
                edge_in_features = edge_in_features.transpose(1, 2)
            edge_in_features = getattr(self, 'activation' + str(i))(edge_in_features)
            edge_in_features = getattr(self, 'dropout' + str(i))(edge_in_features)
        return edge_in_features

class Multi_EGAT(pl.LightningModule):
    def __init__(self, 
                 node_gat_parm: list,
                 edge_gat_parm: list,
                 gat_heads_parm: list,
                 dropout: float,
                 mode: str):
        super().__init__()
        self.node_gat_parm = node_gat_parm
        self.mode = mode
        for i in range(len(node_gat_parm) - 1):
            setattr(self, 'e_gat' + str(i), EdgeGraphAttention(node_gat_parm[i], edge_gat_parm[i], node_gat_parm[i+1], edge_gat_parm[i+1], gat_heads_parm[i], dropout = dropout))
            setattr(self, 'bn_node' + str(i), torch.nn.BatchNorm1d(node_gat_parm[i+1]))
            setattr(self, 'bn_edge' + str(i), torch.nn.BatchNorm1d(edge_gat_parm[i+1]))
            setattr(self, 'activation' + str(i), torch.nn.LeakyReLU())
            setattr(self, 'dropout' + str(i), torch.nn.Dropout(dropout))
    
    def forward(self, node_in_features, edge_in_features, adj_mat):
        for i in range(len(self.node_gat_parm) - 1):
            node_in_features, edge_in_features = getattr(self, 'e_gat' + str(i))(node_in_features, edge_in_features, adj_mat)
            try:
                node_in_features = getattr(self, 'bn_node' + str(i))(node_in_features)
                edge_in_features = getattr(self, 'bn_edge' + str(i))(edge_in_features)
            except:
                pass
            node_in_features = getattr(self, 'activation' + str(i))(node_in_features)
            edge_in_features = getattr(self, 'activation' + str(i))(edge_in_features)
            node_in_features = getattr(self, 'dropout' + str(i))(node_in_features)
            edge_in_features = getattr(self, 'dropout' + str(i))(edge_in_features)
        return node_in_features, edge_in_features

class Multi_GAT(pl.LightningModule):
    def __init__(self, 
                 node_gat_parm: list,
                #  edge_gat_parm: list,
                 gat_heads_parm: list,
                 dropout: float,
                 mode: str):
        super().__init__()
        self.node_gat_parm = node_gat_parm
        self.mode = mode
        for i in range(len(node_gat_parm) - 1):
            setattr(self, 'gat' + str(i), GraphAttentionLayer(node_gat_parm[i], node_gat_parm[i+1], gat_heads_parm[i], dropout = dropout))
            setattr(self, 'bn_node' + str(i), torch.nn.BatchNorm1d(node_gat_parm[i+1]))
            setattr(self, 'activation' + str(i), torch.nn.LeakyReLU())
            setattr(self, 'dropout' + str(i), torch.nn.Dropout(dropout))
    
    def forward(self, node_in_features, adj_mat):
        for i in range(len(self.node_gat_parm) - 1):
            node_in_features = getattr(self, 'gat' + str(i))(node_in_features, adj_mat)
            try:
                node_in_features = getattr(self, 'bn_node' + str(i))(node_in_features)
            except:
                pass
            node_in_features = getattr(self, 'activation' + str(i))(node_in_features)
            node_in_features = getattr(self, 'dropout' + str(i))(node_in_features)
        return node_in_features


class Multi_Readout(pl.LightningModule):
    def __init__(self, 
                 readout_parm: list,
                 dropout: float):
        super().__init__()
        self.readout_parm = readout_parm
        for i in range(len(readout_parm) - 2):
            setattr(self, 'readout' + str(i), torch.nn.Linear(readout_parm[i], readout_parm[i+1]))
            setattr(self, 'bn' + str(i), torch.nn.BatchNorm1d(readout_parm[i+1]))
            setattr(self, 'activation' + str(i), torch.nn.LeakyReLU())
            setattr(self, 'dropout' + str(i), torch.nn.Dropout(dropout))
        setattr(self, 'readout' + str(len(readout_parm) - 2), torch.nn.Linear(readout_parm[len(readout_parm) - 2], readout_parm[len(readout_parm) - 1]))
        setattr(self, 'bn' + str(len(readout_parm) - 2), torch.nn.BatchNorm1d(readout_parm[len(readout_parm) - 1]))
        setattr(self, 'softmax', torch.nn.Softmax(dim=-1))
    
    def forward(self, in_features):
        for i in range(len(self.readout_parm) - 2):

            in_features = getattr(self, 'readout' + str(i))(in_features)
            try:
                in_features = getattr(self, 'bn' + str(i))(in_features)
            except:
                pass
            in_features = getattr(self, 'activation' + str(i))(in_features)
            in_features = getattr(self, 'dropout' + str(i))(in_features)
        in_features = getattr(self, 'readout' + str(len(self.readout_parm) - 2))(in_features)
        try:
            in_features = getattr(self, 'bn' + str(len(self.readout_parm) - 2))(in_features)
        except:
            pass
        in_features = self.softmax(in_features)
        return in_features
    

class MainModel(pl.LightningModule):
    def __init__(
            self,
            node_input_size: int,
            edge_input_size: int,
            edge_emb_parm: list,
            edge_gat_parm: list,
            node_gat_parm: list,
            gat_heads_parm: list,
            node_readout: list,
            edge_readout: list,
            node_class_nb: int,
            edge_class_nb: int,
            dropout: float,
            mode: str,
            ) -> None:
        super().__init__()
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=2)
        self.activation = torch.nn.LeakyReLU()
        self.mode = mode
        print('mode:' + mode)

        if mode == 'pre_train':
            self.node_emb = XceptionTime(node_input_size, node_class_nb)
            self.edge_emb = Edge_emb(edge_emb_parm, dropout)
        elif mode == 'GAT':
            self.node_emb = XceptionTime(2, node_gat_parm[0])
            self.gat = Multi_GAT(node_gat_parm, gat_heads_parm, dropout, mode)
            self.readout_node = Multi_Readout(node_readout, dropout)
            # self.readout_edge = Multi_Readout(edge_readout, dropout)
        elif mode == 'XceptionTime':
            self.node_emb = XceptionTime(2, node_gat_parm[0])
            self.edge_emb = Edge_emb(edge_emb_parm, dropout)
            self.edge_gat = Multi_EGAT(node_gat_parm, edge_gat_parm, gat_heads_parm, dropout, mode)
            self.readout_node = Multi_Readout(node_readout, dropout)
            self.readout_edge = Multi_Readout(edge_readout, dropout)
        elif mode =='BiLSTM':
            self.node_emb = LSTM(2, node_gat_parm[0], bidirectional=False)
            self.edge_emb = Edge_emb(edge_emb_parm, dropout)
            self.edge_gat = Multi_EGAT(node_gat_parm, edge_gat_parm, gat_heads_parm, dropout, mode)
            self.readout_node = Multi_Readout(node_readout, dropout)
            self.readout_edge = Multi_Readout(edge_readout, dropout)
        elif mode == 'Transformer':
            self.node_emb = TransformerModel(2, node_gat_parm[0])
            self.edge_emb = Edge_emb(edge_emb_parm, dropout)
            self.edge_gat = Multi_EGAT(node_gat_parm, edge_gat_parm, gat_heads_parm, dropout, mode)
            self.readout_node = Multi_Readout(node_readout, dropout)
            self.readout_edge = Multi_Readout(edge_readout, dropout)
        # for param in self.readout_edge.parameters():
        #     param.requires_grad = False


            # self.edge_gat1 = EdgeGraphAttention(node_gat_input_size, edge_gat_input_size, node_gat_hidden_size, edge_gat_hidden_size, gat_n_heads, dropout = dropout)
            # self.edge_gat2 = EdgeGraphAttention(node_gat_hidden_size, edge_gat_hidden_size, node_gat_output_size, edge_gat_output_size, 1, is_concat=False, dropout = dropout)

            # self.readout_edge = Readout(edge_gat_output_size * 2, edge_class_nb)
            # self.readout_node = Readout(node_gat_output_size, node_class_nb)

            # self.initialize_weights()

        # print('node_emb', get_parameter_number(self.node_emb))
        # print('edge_emb', get_parameter_number(self.edge_emb))
        # print('edge_gat1', get_parameter_number(self.edge_gat1))
        # print('edge_gat2', get_parameter_number(self.edge_gat2))
        # print('readout_node', get_parameter_number(self.readout_node))
        # print('readout_edge', get_parameter_number(self.readout_edge))

    def initialize_weights(self):
        for m in [self.edge_gat1, self.edge_gat2, self.readout_node, self.readout_edge, self.edge_emb]:
            for name, param in m.named_parameters():
                print(name)
                if 'weight' in name:
                    torch.nn.init.kaiming_uniform_(param)

    def forward(self, node_in_features, edge_in_features, adj_mat):

        node_emb_feat = self.node_emb(node_in_features.permute(0,2,1))
        edge_emb_feat = self.edge_emb(edge_in_features)
        if self.mode == 'GAT':
            node_gat_feat = self.gat(node_emb_feat, adj_mat)
            node_readout = self.readout_node(node_gat_feat)
            return node_readout
        else:
            node_gat_feat, edge_gat_feat = self.edge_gat(node_emb_feat, edge_emb_feat, adj_mat)
                
            edge_gat_feat = edge_gat_feat.reshape(node_gat_feat.shape[0] , node_gat_feat.shape[0], edge_gat_feat.shape[1])
            edge_gat_feat_t = edge_gat_feat.transpose(0, 1)
            edge_gat_feat_concat = torch.cat([edge_gat_feat, edge_gat_feat_t], dim=-1).reshape(node_gat_feat.shape[0] * node_gat_feat.shape[0], edge_gat_feat.shape[-1] * 2)

            node_readout = self.readout_node(node_gat_feat)
            edge_readout = self.readout_edge(edge_gat_feat_concat)

            return node_readout, edge_readout
