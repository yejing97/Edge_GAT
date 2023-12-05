import torch
import os
import numpy as np
import math

class CROHMEDataset(torch.utils.data.Dataset):
    def __init__(self, data_type, root_path, batch_size, max_node, random_padding_size):
        super().__init__()
        print('random_padding_size: ', data_type, random_padding_size)
        self.max_node = max_node
        self.pad = random_padding_size
        self.data_type = data_type
        self.data_path = os.path.join(root_path, self.data_type)
        self.batch_size = batch_size
        self.data_list = os.listdir(self.data_path)
        # self.group_list, self.batch_node_nb, self.batch_edge_nb = self.make_group_list()
        self.sub_eq_list = self.make_list()
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
    
    def make_list(self):
        sub_eq_list = []
        for name in self.data_list:
            n = name.split('_')[0]
            node_nb = int(n.split('E')[0].split('N')[1])
            if node_nb > self.max_node:
                sub_eq_list = sub_eq_list + self.padding(name, self.max_node, node_nb, self.pad)
            else:
                sub_eq_list = sub_eq_list + [[name, node_nb, 0, node_nb]]
        return sub_eq_list

    def padding(self, name, max_node, shape, pad):
        list = []
        list.append([name, shape, 0, max_node - pad])
        start = max_node - pad
        nb = math.ceil((shape + pad) / max_node)
        for i in range(nb - 1):
            if i == nb - 2:
                end = shape
            else:
                end = start + max_node
            list.append([name, shape, start, end])
            start += max_node
        return list
    
    def __getitem__(self, index):
        name, node_nb, start, end = self.sub_eq_list[index]
        data = np.load(os.path.join(self.data_path, name))
        los = torch.from_numpy(data['los'])[start:end,start:end].long()
        am = torch.zeros((los.shape[0], los.shape[1]), dtype=int)
        for i in range(los.shape[0] - 1):
            am[i][i+1] = 1
            am[i+1][i] = 1
        los = am
        if end - start == self.max_node:
            strokes_emb = torch.from_numpy(data['strokes_emb'])[start:end,:,:].float()
            # strokes_emb = self.normalize_gaussian(strokes_emb)
            edges_emb = torch.from_numpy(data['edges_emb'])[start:end,start:end,:,:].float()
            edges_emb = self.normalize_gaussian(edges_emb)
            stroke_labels = torch.from_numpy(data['stroke_labels'])[start:end].long()
            edge_labels = torch.from_numpy(data['edge_labels'])[start:end,start:end].long()
        elif(start == 0 and node_nb > self.max_node):
            padding_stroke = torch.zeros((self.max_node - end, self.node_emb_nb, 2))
            padding_edge = torch.zeros((self.max_node, self.max_node, 4, self.rel_emb_nb))
            padding_stroke_label = torch.zeros((self.max_node - end)).long()
            padding_edge_label = torch.zeros((self.max_node, self.max_node)).long()
            padding_los = torch.zeros((self.max_node, self.max_node)).long()
            strokes_emb = torch.cat((padding_stroke, torch.from_numpy(data['strokes_emb'])[start:end,:,:].float()), dim=0)
            # strokes_emb = self.normalize_gaussian(strokes_emb)
            edges_emb = torch.from_numpy(data['edges_emb'])[start:end,start:end,:,:].float()
            edges_emb = self.normalize_gaussian(edges_emb)
            padding_edge[start + self.pad:,start + self.pad:,:,:] = edges_emb
            edges_emb = padding_edge
            stroke_labels = torch.cat((padding_stroke_label, torch.from_numpy(data['stroke_labels'])[start:end].long()), dim=0)
            edge_labels = torch.from_numpy(data['edge_labels'])[start:end,start:end].long()
            padding_edge_label[start + self.pad:,start + self.pad:] = edge_labels
            edge_labels = padding_edge_label
            # los = torch.from_numpy(data['los'])[start:end,start:end].long()
            padding_los[start + self.pad:,start + self.pad:] = los
            los = padding_los
        elif(end == node_nb or node_nb <= self.max_node):
            padding_stroke = torch.zeros((self.max_node - (end-start), self.node_emb_nb, 2))
            padding_edge = torch.zeros((self.max_node, self.max_node, 4, self.rel_emb_nb))
            padding_stroke_label = torch.zeros(self.max_node - (end-start)).long()
            padding_edge_label = torch.zeros((self.max_node, self.max_node)).long()
            padding_los = torch.zeros((self.max_node, self.max_node)).long()
            strokes_emb = torch.cat((torch.from_numpy(data['strokes_emb'])[start:end,:,:].float(), padding_stroke), dim=0)
            # strokes_emb = self.normalize_gaussian(strokes_emb)
            edges_emb = torch.from_numpy(data['edges_emb'])[start:end,start:end,:,:].float()
            edges_emb = self.normalize_gaussian(edges_emb)
            padding_edge[:end - start,:end - start,:,:] = edges_emb
            edges_emb = padding_edge
            stroke_labels = torch.cat((torch.from_numpy(data['stroke_labels'])[start:end].long(), padding_stroke_label), dim=0)
            edge_labels = torch.from_numpy(data['edge_labels'])[start:end,start:end].long()
            padding_edge_label[:end - start,:end - start] = edge_labels
            edge_labels = padding_edge_label
            # los = torch.from_numpy(data['los'])[start:end,start:end].long()
            padding_los[:end - start,:end - start] = los
            los = padding_los
        else:
            print('error' + name)
        return strokes_emb, edges_emb, los, stroke_labels, edge_labels


    def normalize_gaussian(self, data):
        mean = data.mean(dim = (-2, -1), keepdim=True)
        std = data.std(dim = (-2, -1), keepdim=True)
        return (data - mean) / (std + 1e-7)

    # def __getitem__(self, index):
    #     batch_list = self.group_list[index]
    #     batch_strokes_emb = torch.zeros((self.batch_node_nb[index], self.node_emb_nb, 2))
    #     batch_edges_emb = torch.zeros((self.batch_node_nb[index], self.batch_node_nb[index], 4, self.rel_emb_nb))
    #     batch_stroke_labels = torch.zeros((self.batch_node_nb[index]))
    #     batch_edge_labels = torch.zeros(self.batch_node_nb[index], self.batch_node_nb[index])
    #     batch_los = torch.zeros((self.batch_node_nb[index], self.batch_node_nb[index]))
    #     start = 0
    #     for i in range(len(batch_list)):
    #         data = np.load(os.path.join(self.data_path, self.data_list[batch_list[i]]))
    #         strokes_emb = torch.from_numpy(data['strokes_emb']).float()
    #         batch_strokes_emb[start:start+strokes_emb.shape[0], :, :] = strokes_emb
    #         edges_emb = torch.from_numpy(data['edges_emb']).float()
    #         batch_edges_emb[start:start+edges_emb.shape[0], start:start+edges_emb.shape[1], :, :] = edges_emb
    #         stroke_labels = torch.from_numpy(data['stroke_labels']).long()
    #         batch_stroke_labels[start:start+stroke_labels.shape[0]] = stroke_labels
    #         edge_labels = torch.from_numpy(data['edge_labels']).long()
    #         # edge_labels = torch.where(edge_labels > 1, torch.zeros_like(edge_labels), edge_labels)
    #         batch_edge_labels[start:start+edge_labels.shape[0], start:start+edge_labels.shape[1]] = edge_labels
    #         los = torch.from_numpy(data['los']).long()
    #         batch_los[start:start+los.shape[0], start:start+los.shape[1]] = los
    #         start = start + strokes_emb.shape[0]
    #     # label = self.edge_filter(batch_edge_labels.reshape(-1), batch_los)
    #     return batch_strokes_emb, batch_edges_emb.reshape(self.batch_node_nb[index], self.batch_node_nb[index], edges_emb.shape[2]*edges_emb.shape[3]), batch_los, batch_stroke_labels.long(), batch_edge_labels
    
    # def __len__(self):
    #     return len(self.group_list)

    def __len__(self):
        return len(self.sub_eq_list)


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
    
    def edge_filter(self, edges_label, los):
        los = los.squeeze().fill_diagonal_(0)
        los = torch.triu(los)
        indices = torch.nonzero(los.reshape(-1)).squeeze()
        edges_label = edges_label[indices]
        # edges_emb = edges_emb.reshape(-1, edges_emb.shape[-1])[indices]
        return edges_label
    

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