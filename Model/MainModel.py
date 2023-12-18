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
        self.d = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=2)
        # self.bn_node_1 = torch.nn.BatchNorm1d(node_gat_hidden_size)
        # self.bn_node_2 = torch.nn.BatchNorm1d(node_gat_output_size)
        # self.bn_edge_0 = torch.nn.BatchNorm1d(edge_gat_input_size)
        # self.bn_edge_1 = torch.nn.BatchNorm1d(edge_gat_hidden_size)
        # self.bn_edge_2 = torch.nn.BatchNorm1d(edge_gat_output_size)
        self.bn_node_0 = torch.nn.BatchNorm1d(node_gat_input_size)
        self.bn_node_1 = torch.nn.BatchNorm2d(node_gat_hidden_size)
        self.bn_node_2 = torch.nn.BatchNorm2d(node_gat_output_size)
        self.bn_edge_0 = torch.nn.BatchNorm2d(edge_gat_input_size)
        self.bn_edge_1 = torch.nn.BatchNorm2d(edge_gat_hidden_size)
        self.bn_edge_2 = torch.nn.BatchNorm2d(edge_gat_output_size)

        self.activation = torch.nn.LeakyReLU()

        if self.mode == 'pre_train_node' or self.mode == 'pre_train_edge':
            self.node_emb = XceptionTime(node_input_size, node_class_nb)
            self.edge_emb = torch.nn.Linear(edge_input_size, edge_class_nb)
        else:
            self.node_emb = XceptionTime(node_input_size, node_gat_input_size)
            self.edge_emb = torch.nn.Linear(edge_input_size, edge_gat_input_size)


            self.edge_gat1 = EdgeGraphAttention(node_gat_input_size, edge_gat_input_size, node_gat_hidden_size, edge_gat_hidden_size, gat_n_heads, dropout = dropout)
            self.edge_gat2 = EdgeGraphAttention(node_gat_hidden_size, edge_gat_hidden_size, node_gat_output_size, edge_gat_output_size, 1, is_concat=False, dropout = dropout)

            self.bi_clf = Readout(edge_gat_output_size * 2, 2)
            self.readout_edge = Readout(edge_gat_output_size * 2, edge_class_nb)
            self.readout_node = Readout(node_gat_output_size, node_class_nb)

            # self.initialize_weights()


    def initialize_weights(self):
        for m in [self.edge_gat1, self.edge_gat2, self.readout_node, self.readout_edge, self.edge_emb]:
            for name, param in m.named_parameters():
                print(name)
                if 'weight' in name:
                    torch.nn.init.kaiming_uniform_(param)

    def forward(self, node_in_features, edge_in_features, adj_mat):
        batch_size = node_in_features.shape[0]
        n_node = node_in_features.shape[1]
        node_emb_nb = node_in_features.shape[2]
        # unzip_node, unzip_edge, unzip_adj_mat = self.unzip_batch(node_in_features, edge_in_features, adj_mat)
        if self.mode == 'pre_train_node':
            return self.node_emb(node_in_features.squeeze(0))
        elif self.mode == 'pre_train_edge':
            return self.edge_emb(node_in_features.squeeze(0))
        elif self.mode == 'train':
            node_emb_feat = self.node_emb(node_in_features.reshape(batch_size * n_node , node_emb_nb , -1)).reshape(batch_size * n_node, -1)
            node_emb_feat = self.bn_node_0(node_emb_feat).reshape(batch_size, n_node, -1)
            edge_emb_feat = self.edge_emb(edge_in_features)
            # print(edge_emb_feat.shape)
            # try:
            edge_emb_feat = edge_emb_feat.reshape(batch_size, -1, n_node, n_node)
            edge_emb_feat = self.bn_edge_0(edge_emb_feat).reshape(batch_size, n_node, n_node, -1)

            edge_emb_feat = self.activation(edge_emb_feat)
            node_gat_feat, edge_gat_feat = self.edge_gat1(node_emb_feat, edge_emb_feat, adj_mat)
            node_gat_feat = node_gat_feat.reshape(batch_size, -1, n_node, 1)
            node_gat_feat = self.bn_node_1(node_gat_feat).reshape(batch_size, n_node, -1)
            edge_gat_feat = edge_gat_feat.reshape(batch_size, -1, n_node, n_node)
            edge_gat_feat = self.bn_edge_1(edge_gat_feat).reshape(batch_size, n_node, n_node, -1)
            node_gat_feat = self.activation(node_gat_feat)
            edge_gat_feat = self.activation(edge_gat_feat)
            node_gat_feat, edge_gat_feat = self.edge_gat2(node_gat_feat, edge_gat_feat, adj_mat)
            node_gat_feat = node_gat_feat.reshape(batch_size,-1, n_node, 1)
            node_gat_feat = self.bn_node_2(node_gat_feat).reshape(batch_size, n_node, -1)
            edge_gat_feat = edge_gat_feat.reshape(batch_size, -1, n_node, n_node)
            edge_gat_feat = self.bn_edge_2(edge_gat_feat).reshape(batch_size, n_node, n_node, -1)
            node_gat_feat = self.activation(node_gat_feat)
            edge_gat_feat = self.activation(edge_gat_feat)
            
            edge_gat_feat = edge_gat_feat.reshape(batch_size, n_node, n_node, -1)
            edge_t = edge_gat_feat.transpose(1, 2)
            edge_feat_concat = torch.cat([edge_gat_feat, edge_t], dim=-1)

            edge_feat_concat = edge_feat_concat.reshape(batch_size*n_node*n_node, -1)
            node_gat_feat = node_gat_feat.reshape(batch_size, n_node, -1)
            # edge_readout = self.readout_edge(edge_feat_concat)
            edge_biclf = self.bi_clf(edge_feat_concat)
            edge_biclf = edge_biclf.reshape(batch_size, n_node, n_node, -1)
            edge_feat_concat = edge_feat_concat.reshape(batch_size, n_node, n_node, -1)
            # node_readout = self.readout_node(node_gat_feat)

            node_pooled_feat, edge_pooled_feat = self.sub_graph_pooling(node_gat_feat, edge_feat_concat, edge_biclf)
            # print('node_readout', node_readout.shape)
            # print('edge_readout', edge_readout.shape)
            node_readout = self.readout_node(node_pooled_feat.reshape(batch_size*n_node, -1))
            edge_readout = self.readout_edge(edge_pooled_feat.reshape(batch_size*n_node*n_node, -1))



        return node_readout.reshape(batch_size, n_node, -1), edge_readout.reshape(batch_size, n_node, n_node, -1), edge_biclf.reshape(batch_size, n_node, n_node, -1)
    def unzip_batch(self, strokes_emb, edges_emb, los):
        # strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        # strokes_emb = strokes_emb.squeeze(0)
        strokes_emb = strokes_emb.reshape(strokes_emb.shape[0]*strokes_emb.shape[1], strokes_emb.shape[2], strokes_emb.shape[3])
        edges_emb = edges_emb.reshape(edges_emb.shape[0],edges_emb.shape[1], edges_emb.shape[2], -1)
        # strokes_label = strokes_label.long().reshape(-1)
        # edges_label = edges_label.long()
        # los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)
        los = los + torch.eye(los.shape[1], los.shape[2]).repeat(los.shape[0], 1, 1).to(self.d)
        new_los = torch.zeros((los.shape[0]*los.shape[1], los.shape[0]*los.shape[2])).to(self.d)
        # new_edges_label = torch.zeros((edges_label.shape[0]*edges_label.shape[1], edges_label.shape[0]*edges_label.shape[2])).long().to(self.d)
        new_edges_emb = torch.zeros((edges_emb.shape[0]*edges_emb.shape[1], edges_emb.shape[0]*edges_emb.shape[2], edges_emb.shape[3])).to(self.d)
        for i in range(los.shape[0]):
            new_los[i*los.shape[1]:(i+1)*los.shape[1], i*los.shape[1]:(i+1)*los.shape[1]] = los[i]
            # new_edges_label[i*edges_label.shape[1]:(i+1)*edges_label.shape[1], i*edges_label.shape[1]:(i+1)*edges_label.shape[1]] = edges_label[i]
            new_edges_emb[i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1], i*edges_emb.shape[1]:(i+1)*edges_emb.shape[1]] = edges_emb[i]
        return strokes_emb.to(self.d), new_edges_emb.to(self.d), new_los.unsqueeze(-1).to(self.d)

    def connected_components_mask(self, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.cpu()
        batch_nb = adjacency_matrix.shape[0]
        num_nodes = adjacency_matrix.shape[1]
        visited = torch.zeros(batch_nb, num_nodes, dtype=bool)
        batch_mask = []
        def dfs(batch, node, component):
            visited[batch, node] = True
            component.append(node)
            for neighbor in range(num_nodes):
                if adjacency_matrix[batch][node][neighbor] == 1 and not visited[batch][neighbor]:
                    dfs(batch, neighbor, component)
        for batch in range(batch_nb):
            components = []
            for node in range(num_nodes):
                if not visited[batch, node]:
                    component = []
                    dfs(batch, node, component)
                    components.append(component)
            max_node = max(max(component) for component in components) + 1
            mask = torch.zeros(len(components), max_node, dtype=torch.int)
            for i, component in enumerate(components):
                for node in component:
                    mask[i][node] = 1
            batch_mask.append(mask)
        return batch_mask

    def sub_graph_pooling(self, node_feat, edge_feat, edge_readout):
        batch_size = node_feat.shape[0]
        n_node = node_feat.shape[1]

        cc_graph = torch.argmax(edge_readout, dim=-1).reshape(batch_size, n_node, n_node)
        cc_graph = torch.where(cc_graph == 1, 1, 0)
        batch_mask = self.connected_components_mask(cc_graph)
        batch_node_out = torch.zeros_like(node_feat).to(self.d)
        batch_edge_out = torch.zeros_like(edge_feat).to(self.d)
        for i in range(len(batch_mask)):
            mask = batch_mask[i].to(self.d)
            node_feat_repeat = node_feat[i].unsqueeze(0).repeat(mask.shape[0], 1,1)
            avg_pooled_lines = torch.sum(node_feat_repeat * mask.unsqueeze(-1), dim=1)/ mask.sum(dim=1).unsqueeze(-1)
            node_out = torch.matmul(avg_pooled_lines.t().float(), mask.float()).t()

            edge_feat_repeat = edge_feat[i].unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
            avg_pooled_lines = torch.sum(edge_feat_repeat * mask.unsqueeze(1).unsqueeze(-1), dim=2)/ mask.sum(dim=1).unsqueeze(1).unsqueeze(-1)
            edge_out = torch.matmul(avg_pooled_lines.permute(2,1,0).unsqueeze(1), mask.float().t().unsqueeze(2))

            edge_out_repeat = edge_out.squeeze(-1).permute(2,1,0).unsqueeze(0).repeat(mask.shape[0],1, 1, 1)
            avg_pooled_lines = torch.sum(edge_out_repeat * mask.unsqueeze(-1).unsqueeze(-1), dim=2)/ mask.sum(dim=1).unsqueeze(1).unsqueeze(-1)
            edge_out = torch.matmul(avg_pooled_lines.permute(2,1,0).unsqueeze(1), mask.float().t().unsqueeze(2))
            
            batch_node_out[i] = node_out
            batch_edge_out[i] = edge_out.squeeze(-1).permute(2,1,0)

        return batch_node_out, batch_edge_out

