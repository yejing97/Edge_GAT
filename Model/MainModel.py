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
            # node_emb_size: int,
            # edge_emb_size: int,
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
        # self.node_emb_size = node_emb_size
        # self.edge_input_size = edge_emb_size 
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



        # self.edge_gat = EdgeGraphAttention(self.gat_input_size, self.gat_output_size, self.gat_n_heads, self.dropout)

        self.gat1 = GraphAttentionLayer(self.gat_input_size, self.gat_hidden_size, self.gat_n_heads, dropout=self.dropout)

        self.gat2 = GraphAttentionLayer(self.gat_hidden_size, self.gat_output_size, 1, is_concat=False, dropout=self.dropout)

        # self.readout_node = Readout(self.gat_output_size, self.node_class_nb)
        self.readout_edge = Readout(self.gat_output_size, self.edge_class_nb)

        self.readout_node = torch.nn.Linear(self.gat_input_size, self.node_class_nb)

        print('node_emb', get_parameter_number(self.node_emb))
        print('edge_emb', get_parameter_number(self.edge_emb))
        # print('edge_gat', get_parameter_number(self.edge_gat))
        print('gat1', get_parameter_number(self.gat1))
        print('gat2', get_parameter_number(self.gat2))
        print('readout_node', get_parameter_number(self.readout_node))
        print('readout_edge', get_parameter_number(self.readout_edge))

    def forward(self, node_in_features, edge_in_features, adj_mat):
        node_emb_feat = self.node_emb(node_in_features.squeeze(0))
        # edge_emb_feat = self.edge_emb(edge_in_features.squeeze(0))

        # node_gat_feat, edge_gat_feat = self.edge_gat(node_emb_feat, edge_emb_feat, adj_mat.squeeze(0))
        

        # indices = torch.nonzero(adj_mat.reshape(-1)).squeeze()
        # print('pred',indices)
        # edge_gat_feat = edge_gat_feat.reshape(-1, self.gat_output_size)[indices]

        # node_gat_feat = self.gat1(node_emb_feat, adj_mat.unsqueeze(-1))
        # node_gat_feat = self.activation(node_gat_feat)
        # node_gat_feat = self.gat2(node_gat_feat, adj_mat.unsqueeze(-1))

        # node_readout = self.readout_node(node_emb_feat)
        # edge_readout = self.readout_edge(edge_gat_feat)
        return node_emb_feat, None
        # return node_readout, edge_readout