import torch
import tsai
# from labml_helpers.module import Module
import pytorch_lightning as pl
from EdgeGat import EdgeGraphAttention, Readout

class MainModel(pl.LightningModule):
    def __init__(
            self,
            node_input_size: int,
            edge_input_size: int,
            node_emb_size: int,
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
        self.node_emb_size = node_emb_size
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

        self.node_emb = tsai.all.XceptionModule(self.node_input_size, self.gat_input_size)
        self.edge_emb = torch.nn.Linear(self.edge_input_size, self.gat_input_size)

        self.edge_gat = EdgeGraphAttention(self.gat_input_size, self.gat_hidden_size, self.gat_output_size, self.gat_n_heads, self.dropout)
        self.readout_node = Readout(self.gat_output_size, self.node_class_nb)
        self.readout_edge = Readout(self.gat_output_size, self.edge_class_nb)

    def forward(self, node_in_features, edge_in_features, adj_mat):
        node_emb_feat = self.node_emb(node_in_features)
        edge_emb_feat = self.edge_emb(edge_in_features)
        node_gat_feat, edge_gat_feat = self.edge_gat(node_emb_feat, edge_emb_feat, adj_mat)
        node_readout = self.readout_node(node_gat_feat)
        edge_readout = self.readout_edge(edge_gat_feat)
        return node_readout, edge_readout