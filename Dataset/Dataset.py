import torch
import os
import numpy as np

class CROHMEDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, root_path, batch_size, max_node):
        super().__init__()
        self.data_type = data_type
        self.data_path = os.path.join(root_path, self.data_type)
        self.batch_size = batch_size
        self.data_list = self.make_data_list(max_node)
        self.group_list, self.batch_node_nb, self.batch_edge_nb = self.make_group_list()
        self.node_emb_nb = int(root_path.split('/')[-1].split('_')[0].split('S')[1])
        self.rel_emb_nb = int(root_path.split('/')[-1].split('_')[1].split('R')[1])

    def make_data_list(self, max_node):
        if max_node == -1:
            return os.listdir(self.data_path)
        else:
            data_list = []
            for name in os.listdir(self.data_path):
                if int(name.split('E')[0].split('N')[1]) <= max_node:
                    data_list.append(name)
            return data_list

    def make_group_list(self):
        node_nb = []
        edge_nb = []
        for name in self.data_list:
            n = name.split('_')[0]
            node_nb.append(int(n.split('E')[0].split('N')[1]))
            edge_nb.append(int(n.split('E')[1]))
        group_list = self.balance_graphs(node_nb, edge_nb, self.batch_size)
        # sum node for every group
        batch_node_nb = []
        batch_edge_nb = []
        for list in group_list:
            batch_node_nb.append(sum([int(node_nb[i]) for i in list]))
            batch_edge_nb.append(sum([int(edge_nb[i]) for i in list]))
        # sum edge for every group
        return group_list, batch_node_nb, batch_edge_nb

    def __getitem__(self, index):
        batch_list = self.group_list[index]
        batch_strokes_emb = torch.zeros((self.batch_node_nb[index], self.node_emb_nb, 2))
        batch_edges_emb = torch.zeros((self.batch_node_nb[index], self.batch_node_nb[index], 4, self.rel_emb_nb))
        batch_stroke_labels = torch.zeros((self.batch_node_nb[index]))
        batch_edge_labels = torch.zeros(self.batch_node_nb[index], self.batch_node_nb[index])
        batch_los = torch.zeros((self.batch_node_nb[index], self.batch_node_nb[index]))
        start = 0
        for i in range(len(batch_list)):
            data = np.load(os.path.join(self.data_path, self.data_list[batch_list[i]]))
            strokes_emb = torch.from_numpy(data['strokes_emb']).float()
            batch_strokes_emb[start:start+strokes_emb.shape[0], :, :] = strokes_emb
            edges_emb = torch.from_numpy(data['edges_emb']).float()
            batch_edges_emb[start:start+edges_emb.shape[0], start:start+edges_emb.shape[1], :, :] = edges_emb
            stroke_labels = torch.from_numpy(data['stroke_labels']).long()
            batch_stroke_labels[start:start+stroke_labels.shape[0]] = stroke_labels
            edge_labels = torch.from_numpy(data['edge_labels']).long()
            batch_edge_labels[start:start+edge_labels.shape[0], start:start+edge_labels.shape[1]] = edge_labels
            los = torch.from_numpy(data['los']).long()
            batch_los[start:start+los.shape[0], start:start+los.shape[1]] = los
            start += strokes_emb.shape[0]
        return batch_strokes_emb, batch_edges_emb.reshape(self.batch_node_nb[index], self.batch_node_nb[index], edges_emb.shape[2]*edges_emb.shape[3]), batch_los, batch_stroke_labels.long(), batch_edge_labels
    
    def __len__(self):
        return len(self.group_list)


    def balance_graphs(self, node_counts, edge_counts, nodes_per_group):
        num_groups = sum(node_counts) // nodes_per_group
        num_graphs = len(node_counts)

        # Calculate the total number of edges
        total_edges = sum(edge_counts)

        # Sort graphs by edge counts in ascending order
        sorted_graphs = sorted(enumerate(edge_counts), key=lambda x: x[1])

        # Initialize group assignments
        group_assignments = [-1] * num_graphs

        # Initialize group statistics
        group_node_sums = [0] * num_groups
        group_edge_sums = [0] * num_groups

        for graph_idx, edge_count in sorted_graphs:
            # Try to add the graph to the group with the smallest sum of nodes
            min_group = group_node_sums.index(min(group_node_sums))
            if group_assignments[graph_idx] == -1:
                if group_node_sums[min_group] + node_counts[graph_idx] <= nodes_per_group:
                    group_assignments[graph_idx] = min_group
                    group_node_sums[min_group] += node_counts[graph_idx]
                    group_edge_sums[min_group] += edge_count

        # If there are unassigned graphs, assign them to the group with the smallest edge sum
        for graph_idx, edge_count in sorted_graphs:
            if group_assignments[graph_idx] == -1:
                min_group = group_edge_sums.index(min(group_edge_sums))
                group_assignments[graph_idx] = min_group
                group_edge_sums[min_group] += edge_count

        # Organize the groups
        groups = [[] for _ in range(num_groups)]
        for graph_idx, group_idx in enumerate(group_assignments):
            groups[group_idx].append(graph_idx)

        return groups

    # def __getitem__(self, index):
    #     data = np.load(os.path.join(self.data_path, self.data_list[index]))
    #     strokes_emb = torch.from_numpy(data['strokes_emb']).float()
    #     edges_emb = torch.from_numpy(data['edges_emb']).float()
    #     stroke_labels = torch.from_numpy(data['stroke_labels']).long()
    #     edge_labels = torch.from_numpy(data['edge_labels']).long()
    #     los = torch.from_numpy(data['los']).long()
    #     return strokes_emb, edges_emb.squeeze().reshape(edges_emb.shape[0], edges_emb.shape[1], edges_emb.shape[2]*edges_emb.shape[3]), los, stroke_labels, edge_labels
    
    # def __len__(self):
    #     return len(self.data_list)