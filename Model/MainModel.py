import torch
# import tsai
from tsai.all import *
# from labml_helpers.module import Module
import pytorch_lightning as pl
from Model.EdgeGat import EdgeGraphAttention, Readout

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class MainModel(pl.LightningModule):
    def __init__(
            self,
            node_input_size: int,
            edge_input_size: int,
            node_gat_input_size: int,
            edge_gat_input_size: int,
            node_gat_hidden_size: int,
            edge_gat_hidden_size: int,
            node_gat_output_size: int,
            edge_gat_output_size: int,
            gat_n_heads: int,
            node_class_nb: int,
            edge_class_nb: int,
            dropout: float,
            mode: str = 'pre_train_node',
            ) -> None:
        super().__init__()
        self.mode = mode
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=2)
        self.bn_edge_0 = torch.nn.BatchNorm1d(edge_gat_input_size)
        self.bn_node_1 = torch.nn.BatchNorm1d(node_gat_hidden_size)
        self.bn_node_2 = torch.nn.BatchNorm1d(node_gat_output_size)
        self.bn_edge_1 = torch.nn.BatchNorm1d(edge_gat_hidden_size)
        self.bn_edge_2 = torch.nn.BatchNorm1d(edge_gat_output_size)
        self.activation = torch.nn.LeakyReLU()

        if self.mode == 'pre_train_node' or self.mode == 'pre_train_edge':
            self.node_emb = XceptionTime(node_input_size, node_class_nb)
            self.edge_emb = torch.nn.Linear(edge_input_size, edge_class_nb)
        else:
            self.node_emb = XceptionTime(node_input_size, node_gat_input_size)
            self.edge_emb = torch.nn.Linear(edge_input_size, edge_gat_input_size)


            # self.edge_gat1 = EdgeGraphAttention(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout = self.dropout)
            self.edge_gat1 = EdgeGraphAttention(node_gat_input_size, edge_gat_input_size, node_gat_hidden_size, edge_gat_hidden_size, gat_n_heads, dropout = dropout)
            self.edge_gat2 = EdgeGraphAttention(node_gat_hidden_size, edge_gat_hidden_size, node_gat_output_size, edge_gat_output_size, 1, is_concat=False, dropout = dropout)
            # self.edge_gat = EdgeGraphAttention(node_gat_input_size, edge_gat_input_size, node_gat_output_size, edge_gat_output_size, gat_n_heads, dropout)

            self.readout_edge = Readout(edge_gat_output_size * 2, edge_class_nb)
            self.readout_node = Readout(node_gat_output_size, node_class_nb)

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
        if self.mode == 'pre_train_node':
            return self.node_emb(node_in_features.squeeze(0))
        elif self.mode == 'pre_train_edge':
            return self.edge_emb(edge_in_features.squeeze(0))
        elif self.mode == 'train':
            node_emb_feat = self.node_emb(node_in_features)
            edge_emb_feat = self.edge_emb(edge_in_features)
            # print(edge_emb_feat.shape)
            try:
                edge_emb_feat = self.bn_edge_0(edge_emb_feat.transpose(1, 2)).transpose(1, 2)
            except:
                edge_emb_feat = edge_emb_feat
                
            edge_emb_feat = self.activation(edge_emb_feat)
            node_gat_feat, edge_gat_feat = self.edge_gat1(node_emb_feat, edge_emb_feat, adj_mat)
            try:
                node_gat_feat = self.bn_node_1(node_gat_feat)
                edge_gat_feat = self.bn_edge_1(edge_gat_feat)
            except:
                node_gat_feat = node_gat_feat
                edge_gat_feat = edge_gat_feat
            node_gat_feat = self.activation(node_gat_feat)
            edge_gat_feat = self.activation(edge_gat_feat)
            # print('node_gat_feat', node_gat_feat.shape)
            # print('edge_gat_feat', edge_gat_feat.shape)
            node_gat_feat, edge_gat_feat = self.edge_gat2(node_gat_feat, edge_gat_feat, adj_mat)
            try:
                node_gat_feat = self.bn_node_2(node_gat_feat)
                edge_gat_feat = self.bn_edge_2(edge_gat_feat)
            except:
                node_gat_feat = node_gat_feat
                edge_gat_feat = edge_gat_feat
            # node_gat_feat = self.bn_node_2(node_gat_feat)
            # edge_gat_feat = self.bn_edge_2(edge_gat_feat)
            node_gat_feat = self.activation(node_gat_feat)
            edge_gat_feat = self.activation(edge_gat_feat)
            
            # node_repeat = node_gat_feat.repeat(1, node_gat_feat.shape[0]).reshape(node_gat_feat.shape[1], node_gat_feat.shape[0], node_gat_feat.shape[0])
            # eye = torch.eye(node_gat_feat.shape[0], node_gat_feat.shape[0]).to(node_gat_feat.device).repeat(node_gat_feat.shape[1], 1, 1)
            # node_out = (node_repeat * eye).reshape(node_gat_feat.shape[0] * node_gat_feat.shape[0], node_gat_feat.shape[1])
            node_readout = self.readout_node(node_gat_feat)
            edge_gat_feat = edge_gat_feat.reshape(node_gat_feat.shape[0] , node_gat_feat.shape[0], edge_gat_feat.shape[1])
            edge_t = edge_gat_feat.transpose(0, 1)

            edge_feat_concat = torch.cat([edge_gat_feat, edge_t], dim=-1)
            edge_readout = self.readout_edge(edge_feat_concat.reshape(edge_feat_concat.shape[0] * edge_feat_concat.shape[1], edge_feat_concat.shape[2]))
        # return node_out, edge_gat_feat
        return node_readout, edge_readout
