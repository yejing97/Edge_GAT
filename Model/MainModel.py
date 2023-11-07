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

class MainModel(pl.LightningModule):
    def __init__(
            self,
            node_input_size: int,
            edge_input_size: int,
            # gat_input_size: int,
            node_gat_input_size: int,
            edge_gat_input_size: int,
            # gat_hidden_size: int,
            node_gat_hidden_size: int,
            edge_gat_hidden_size: int,
            # gat_output_size: int,
            node_gat_output_size: int,
            edge_gat_output_size: int,
            gat_n_heads: int,
            node_class_nb: int,
            edge_class_nb: int,
            dropout: float = 0.6,
            ) -> None:
        super().__init__()
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        # self.gat_input_size = gat_input_size
        # self.gat_hidden_size = gat_hidden_size
        # self.gat_output_size = gat_output_size

        # self.node_gat_input_size = node_gat_input_size
        # self.edge_gat_input_size = edge_gat_input_size
        # self.node_gat_hidden_size = node_gat_hidden_size
        # self.edge_gat_hidden_size = edge_gat_hidden_size
        # self.node_gat_output_size = node_gat_output_size
        # self.edge_gat_output_size = edge_gat_output_size
        # self.gat_n_heads = gat_n_heads
        # self.node_class_nb = node_class_nb
        # self.edge_class_nb = edge_class_nb
        # self.dropout = dropout

        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=2)
        self.activation = torch.nn.LeakyReLU()

        self.node_emb = XceptionTime(node_input_size, node_gat_input_size)
        self.edge_emb = torch.nn.Linear(edge_input_size, edge_gat_input_size)



        # self.edge_gat1 = EdgeGraphAttention(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout = self.dropout)
        self.edge_gat1 = EdgeGraphAttention(node_gat_input_size, edge_gat_input_size, node_gat_hidden_size[0], edge_gat_hidden_size[0], gat_n_heads, dropout = dropout)
        self.edge_gat2 = EdgeGraphAttention(node_gat_hidden_size[0], edge_gat_hidden_size[0], node_gat_hidden_size[1], edge_gat_hidden_size[1], gat_n_heads, dropout = dropout)
        self.edge_gat3 = EdgeGraphAttention(node_gat_hidden_size[1], edge_gat_hidden_size[1], node_gat_hidden_size[2], edge_gat_hidden_size[2], gat_n_heads, dropout = dropout)
        self.edge_gat4 = EdgeGraphAttention(node_gat_hidden_size[2], edge_gat_hidden_size[2], node_gat_hidden_size[3], edge_gat_hidden_size[3], gat_n_heads, dropout = dropout)
        self.edge_gat5 = EdgeGraphAttention(node_gat_hidden_size[3], edge_gat_hidden_size[3], node_gat_hidden_size[4], edge_gat_hidden_size[4], gat_n_heads, dropout = dropout)
        self.edge_gat6 = EdgeGraphAttention(node_gat_hidden_size[4], edge_gat_hidden_size[4], node_gat_output_size, edge_gat_output_size, 1, is_concat=False, dropout = dropout)

        self.readout_edge = Readout(edge_gat_output_size, edge_class_nb)
        self.readout_node = Readout(node_gat_output_size, node_class_nb)

        # print('node_emb', get_parameter_number(self.node_emb))
        # print('edge_emb', get_parameter_number(self.edge_emb))
        # print('edge_gat1', get_parameter_number(self.edge_gat1))
        # print('edge_gat2', get_parameter_number(self.edge_gat2))
        # print('readout_node', get_parameter_number(self.readout_node))
        # print('readout_edge', get_parameter_number(self.readout_edge))

    def forward(self, node_in_features, edge_in_features, adj_mat):
        # print('node_in_features', node_in_features.shape)
        # print('edge_in_features', edge_in_features.shape)
        node_emb_feat = self.node_emb(node_in_features.squeeze(0))
        edge_emb_feat = self.edge_emb(edge_in_features.squeeze(0))
        # print('node_emb_feat', node_emb_feat.shape)
        # print('edge_emb_feat', edge_emb_feat.shape)
        node_gat_feat, edge_gat_feat = self.edge_gat1(node_emb_feat, edge_emb_feat, adj_mat)
        node_gat_feat = self.activation(node_gat_feat)
        edge_gat_feat = self.activation(edge_gat_feat)
        # print('node_gat_feat', node_gat_feat.shape)
        # print('edge_gat_feat', edge_gat_feat.shape)
        node_gat_feat, edge_gat_feat = self.edge_gat2(node_gat_feat, edge_gat_feat, adj_mat)


        node_readout = self.readout_node(node_gat_feat)
        edge_readout = self.readout_edge(edge_gat_feat)
        return node_readout, edge_readout