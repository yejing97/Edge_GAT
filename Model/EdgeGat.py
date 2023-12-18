
import torch
from torch import nn

# from labml_helpers.module import Module
import pytorch_lightning as pl



class EdgeGraphAttention(pl.LightningModule):

    def __init__(self, 
                 node_in_features: int, 
                 edge_in_features: int,
                 node_out_features: int,
                 edge_out_features: int,
                 n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        # self.node_out_features = node_out_features
        # self.edge_out_features = edge_out_features

        if is_concat:
            # assert out_features % n_heads == 0
            # self.n_hidden = out_features // n_heads
            assert node_out_features % n_heads == 0 and edge_out_features % n_heads == 0
            self.n_hidden_node = node_out_features // n_heads
            self.n_hidden_edge = edge_out_features // n_heads
            self.n_hidden_mix = self.n_hidden_node * 2 + self.n_hidden_edge
        else:
            self.n_hidden_node = node_out_features
            self.n_hidden_edge = edge_out_features
            self.n_hidden_mix = self.n_hidden_node * 2 + self.n_hidden_edge
            # self.n_hidden = out_features // 3
        self.linear_l = nn.Linear(node_in_features, self.n_hidden_node * n_heads, bias=False)
        self.linear_b = nn.Linear(edge_in_features, self.n_hidden_edge * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(node_in_features, self.n_hidden_node * n_heads, bias=False)
        # self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.attn = nn.Linear(self.n_hidden_node * 2 + self.n_hidden_edge, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        # self.bn_ll = nn.BatchNorm1d(self.n_hidden_node * n_heads)
        # self.bn_lr = nn.BatchNorm1d(self.n_hidden_node * n_heads)
        # self.bn_lb = nn.BatchNorm1d(self.n_hidden_edge * n_heads)

    def forward(self, h: torch.Tensor, b: torch.Tensor, adj_mat: torch.Tensor):
        n_batch = h.shape[0]
        n_nodes = h.shape[1]

        g_l = self.linear_l(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden_node)
        # g_l = self.bn_ll(g_l).view(n_nodes, self.n_heads, self.n_hidden_node)
        g_r = self.linear_r(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden_node)
        # g_r = self.bn_lr(g_r).view(n_nodes, self.n_heads, self.n_hidden_node)

        g_b = self.linear_b(b).view(n_batch, n_nodes*n_nodes, self.n_heads,self.n_hidden_edge)
        # g_b = self.bn_lb(g_b).view(n_nodes*n_nodes, self.n_heads, self.n_hidden_edge)
        g_l_repeat = g_l.repeat(1,n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)
        g_concat = torch.cat([g_l_repeat, g_r_repeat_interleave, g_b], dim=-1)

        g_concat = g_concat.view(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden_mix)

        # g_sum = g_l_repeat + g_r_repeat_interleave + g_b
        # g_sum = g_l_repeat + g_r_repeat_interleave
        
        # g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # e = self.attn(self.activation(g_sum))
        e = self.attn(self.activation(g_concat))
        e = e.squeeze(-1)

        # add self connection
        # adj_mat_self = adj_mat.fill_diagonal_(1).unsqueeze(-1)
        if len(adj_mat.shape) == 3:
            adj_mat_self = adj_mat.unsqueeze(-1)
        else:
            adj_mat_self = adj_mat

        assert adj_mat_self.shape[1] == 1 or adj_mat_self.shape[1] == n_nodes
        assert adj_mat_self.shape[2] == 1 or adj_mat_self.shape[2] == n_nodes
        assert adj_mat_self.shape[3] == 1 or adj_mat_self.shape[3] == self.n_heads
        # print(adj_mat_self.shape)
        # print(e.shape)
        e = e.masked_fill(adj_mat_self == 0, float(-1e9))


        a = self.softmax(e)

        # a = self.dropout(a)


        #h_{q} = \sum_{i=0}^{N}( \alpha  ^{i,j} \cdot W_{h}\cdot \sum_{j\in \mathcal{N} _{i}} h_{q-1}^{j})
        new_node = torch.einsum('bijh,bjhf->bihf', a, g_r)
        # b_{q}= \sum_{i,j=0}^{N}( \alpha  ^{i,j} \cdot W_{b}\cdot b_{q-1}^{i,j})
        new_edge = torch.einsum('bijh,bijhf->bijhf', a, g_b.view(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden_edge))
        # torch.set_printoptions(profile="full")
        # print(new_edge)
        # if self.is_concat:
        return new_node.reshape(n_batch, n_nodes, self.n_heads * self.n_hidden_node), new_edge.reshape(n_batch, n_nodes * n_nodes, self.n_heads * self.n_hidden_edge)
        # else:
        #     return new_node.reshape(n_nodes, self.n_heads * self.n_hidden).mean(dim=1), new_edge.reshape(n_nodes, n_nodes, self.n_heads * self.n_hidden).mean(dim=2)


class Readout(pl.LightningModule):
    def __init__(self, in_features: int, class_nb: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, class_nb)
        self.batch_norm = nn.BatchNorm1d(class_nb)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        x = self.linear(x)
        # try:
        x = self.batch_norm(x)
        # except:
        #     pass
        # x = self.batch_norm(x)
        x = self.softmax(x)
        return x
    
