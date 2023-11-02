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
            gat_input_size: int,
            gat_hidden_size: int,
            gat_output_size: int,
            gat_n_heads: int,
            node_class_nb: int,
            edge_class_nb: int,
            dropout: float = 0.6,
            ) -> None:
        super().__init__()
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.gat_input_size = gat_input_size
        self.gat_hidden_size = gat_hidden_size
        self.gat_output_size = gat_output_size
        self.gat_n_heads = gat_n_heads
        self.node_class_nb = node_class_nb
        self.edge_class_nb = edge_class_nb
        self.dropout = dropout

        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=2)
        self.activation = torch.nn.ReLU()

        self.node_emb = XceptionTime(self.node_input_size, self.node_class_nb)
        self.edge_emb = torch.nn.Linear(self.edge_input_size, self.gat_input_size)



        self.edge_gat1 = EdgeGraphAttention(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout = self.dropout)
        self.edge_gat2 = EdgeGraphAttention(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout = self.dropout)


        # self.gat1 = GraphAttentionLayer(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout=self.dropout)

        # self.gat2 = GraphAttentionLayer(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout=self.dropout)

        # self.readout_node = Readout(self.gat_output_size, self.node_class_nb)
        self.readout_edge = Readout(self.gat_output_size, self.edge_class_nb)
        self.readout_node = Readout(self.gat_output_size, self.node_class_nb)

        # self.readout_node = torch.nn.Linear(self.gat_input_size, self.node_class_nb)

        print('node_emb', get_parameter_number(self.node_emb))
        print('edge_emb', get_parameter_number(self.edge_emb))
        # print('edge_gat', get_parameter_number(self.edge_gat))
        print('edge_gat1', get_parameter_number(self.edge_gat1))
        print('edge_gat2', get_parameter_number(self.edge_gat2))
        print('readout_node', get_parameter_number(self.readout_node))
        print('readout_edge', get_parameter_number(self.readout_edge))

    def forward(self, node_in_features, edge_in_features, adj_mat):
        print('node_in_features', node_in_features.shape)
        print('edge_in_features', edge_in_features.shape)
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

        # indices = torch.nonzero(adj_mat.reshape(-1)).squeeze()
        # print('pred',indices)
        # edge_gat_feat = edge_gat_feat.reshape(-1, self.gat_output_size)[indices]

        node_readout = self.readout_node(node_gat_feat)
        edge_readout = self.readout_edge(edge_gat_feat)
        return node_readout, edge_readout