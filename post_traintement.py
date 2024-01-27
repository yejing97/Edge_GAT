import torch
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def load_pt(path):
    pt = torch.load(path, map_location=torch.device('cpu'))
    stroke_emb = pt[0]
    edge_emb = pt[1]
    stroke_label = pt[2]
    edge_label = pt[3]
    los = pt[4]
    return stroke_emb, edge_emb, stroke_label, edge_label, los

def remove_extra_class(edge_pred):
    for i in range(edge_pred.shape[0]):
        for j in range(edge_pred.shape[1]):
            if edge_pred[i,j] > 7:
                edge_pred[j,i] = edge_pred[i,j] - 6
    return edge_pred

def connected_components_mask(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.cpu()
        # batch_nb = adjacency_matrix.shape[0]
        num_nodes = adjacency_matrix.shape[0]
        visited = torch.zeros(num_nodes, dtype=bool)
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(num_nodes):
                if adjacency_matrix[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, component)
        components = []
        for node in range(num_nodes):
            if not visited[node]:
                component = []
                dfs(node, component)
                components.append(component)
        max_node = max(max(component) for component in components) + 1
        mask = torch.zeros(len(components), max_node, dtype=torch.int)
        for i, component in enumerate(components):
            for node in component:
                mask[i][node] = 1
        return mask, components

def sub_graph_pooling(node_feat, edge_feat, mask):
    # am = torch.argmax(edge_feat, dim=2)
    # mask, _ = connected_components_mask(am)

    node_feat_repeat = node_feat.unsqueeze(0).repeat(mask.shape[0], 1,1)
    node_out = torch.sum(node_feat_repeat * mask.unsqueeze(-1), dim=1)/ mask.sum(dim=1).unsqueeze(-1)

    edge_out = torch.zeros(mask.shape[0], mask.shape[0], edge_feat.shape[-1])

    # node_out = torch.matmul(avg_pooled_lines.t().float(), mask.float()).t()

    edge_feat_repeat = edge_feat.unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
    avg_pooled_lines = torch.sum(edge_feat_repeat * mask.unsqueeze(1).unsqueeze(-1), dim=2)/ mask.sum(dim=1).unsqueeze(1).unsqueeze(-1)

    # edge_out = torch.matmul(avg_pooled_lines.permute(2,1,0).unsqueeze(1), mask.float().t().unsqueeze(2))
    # print(torch.argmax(avg_pooled_lines,dim=2))
    # print(avg_pooled_lines.shape)
    edge_out_repeat = avg_pooled_lines.squeeze(-1).permute(2,1,0).unsqueeze(0).repeat(mask.shape[0],1, 1, 1)
    edge_out_repeat = edge_out_repeat.permute(0,3,2,1)

    edge_out = torch.sum(edge_out_repeat * mask.unsqueeze(1).unsqueeze(-1), dim=2)/ mask.sum(dim=1).unsqueeze(1).unsqueeze(-1)


    return node_out, edge_out

def stroke2sym(node_label, componets):
    new_label = torch.zeros(len(componets))
    for i in range(len(componets)):
        j = componets[i][0]
        new_label[i] = node_label[j]
    return new_label
def sym2stroke(node_label, componets, nb):
    new_label = torch.zeros(nb)
    for i in range(len(componets)):
        for j in range(len(componets[i])):
            new_label[componets[i][j]] = node_label[i]
    return new_label

def lg2symlg(edge_label,componets):
    for i in range(len(componets)):
        componet = componets[i]
        x = edge_label[:,componet]
        x = torch.where(x==1, 0,x)
        before_x = edge_label[:,:componet[0]]
        after_x = edge_label[:,componet[-1] +1:]
        x_new = torch.max(x,dim=1).values.unsqueeze(-1)
        edge_label = torch.cat((before_x, x_new, after_x), dim=1)
        y = edge_label[componet,:]
        y = torch.where(y==1, 0,y)
        y_new = torch.max(y, dim=0).values.unsqueeze(0)
        before_y = edge_label[:componet[0],:]
        after_y = edge_label[componet[-1]+1 :,:]

        edge_label = torch.cat((before_y, y_new, after_y), dim=0)
        gap = len(componet) - 1
        for j in range(len(componets)):
            for k in range(len(componets[j])):
                componets[j][k] = componets[j][k] - gap
        
    return edge_label

def find_duplicates_except_zero(list):
    seen = {}
    duplicate_indices = []

    for i in range(len(list)):
        if list[i] != 0:
            if list[i]  not in seen:
                seen[list[i]] = i
            else:
                # duplicate_indices[seen[list[i]]] = i
                duplicate_indices.append([seen[list[i]], i])
    grouped_dict = {}
    for sublist in duplicate_indices:
        key = sublist[0]
        value = sublist[1:]
        grouped_dict.setdefault(key, []).extend(value)
    out = []
    for key in grouped_dict:
        out.append([key] + grouped_dict[key])
    return out

def remove_repeat(edge_pred):
    edge_pred = edge_pred.fill_diagonal_(0)
    for i in range(edge_pred.shape[0]):
        repeat_x = find_duplicates_except_zero(edge_pred[i,:].tolist())
        repeat_y = find_duplicates_except_zero(edge_pred[:,i].tolist())
        if repeat_x != []:
            for j in repeat_x:
                near = [abs(x - i) for x in j]
                keep = min(near)
                for x in j:
                    if x != keep:
                        edge_pred[i,x] = 0
        if repeat_y != []:
            for j in repeat_y:
                near = [abs(x - i) for x in j]
                keep = min(near)
                for x in j:
                    if x != keep:
                        edge_pred[x,i] = 0
    return edge_pred

def is_directed_tree(adjacency_matrix):
    num_nodes = adjacency_matrix.size(0)
    
    num_edges = torch.sum(adjacency_matrix)
    condition1 = (num_nodes == num_edges + 1)

    def dfs(node, visited):
        visited[node] = True
        for neighbor in range(num_nodes):
            if adjacency_matrix[node, neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, visited)

    visited = torch.zeros(num_nodes, dtype=torch.bool)

    dfs(0, visited)

    condition2 = visited.all()

    condition3 = (torch.sum(adjacency_matrix) == num_nodes - 1)

    return condition1 and condition2 and condition3

def add_lost(edge_pred):
    for i in range(edge_pred.shape[0] - 1):
        if torch.all(edge_pred[i,:] == 0).item() and torch.all(edge_pred[:,i] == 0).item():
            edge_pred[i,i + 1] = 2
    edge_mask = torch.where(edge_pred == 0, 0, 1)
    if not is_directed_tree(edge_mask):
        for i in range(edge_pred.shape[0] - 1):
            for j in range(i + 1, edge_pred.shape[0]):
                if torch.all(edge_pred[i,:] == 0).item() and torch.all(edge_pred[:,j] == 0).item():
                    edge_pred[i,j] = 2
                    break
    return edge_pred
def edge_filter(edges_pred, edges_label, los):
        # return
        edges_pred = edges_pred.reshape(-1)
        edges_label = edges_label.reshape(-1)

        
        edges_label = torch.where(edges_label < 14, edges_label, 0)
        try:
            los = los.squeeze().fill_diagonal_(0)
            los = torch.triu(los)
        except:
            los = torch.zeros(0)
        indices = torch.nonzero(los.reshape(-1)).squeeze().reshape(-1)
        edges_label = edges_label[indices]
        edges_pred = edges_pred[indices]
        return edges_pred, edges_label
        

if __name__ == '__main__':
    results_path = '/home/e19b516g/yejing/code/Edge_GAT/results/corrected/test'
    edge_correct = 0
    structure_correct = 0
    all_correct = 0
    error = 0
    all_node_pred = torch.zeros(0)
    all_node_label = torch.zeros(0)
    all_edge_pred = torch.zeros(0)
    all_edge_label = torch.zeros(0)
    for root, _, files in os.walk(results_path):
        for file in files:
            if file.endswith('.pt'):
                path = os.path.join(root, file)
                stroke_emb, edge_emb, stroke_label, edge_label, los = load_pt(path)
                am = torch.where(edge_label == 1, 1, 0)
                # am = torch.where(torch.argmax(edge_emb, dim=2) == 1, 1, 0)
                try:
                    mask, componets = connected_components_mask(am)
                    node_out, edge_out = sub_graph_pooling(stroke_emb, edge_emb, mask)
                    edge_label = remove_extra_class(edge_label)
                    edge_pred = torch.argmax(edge_out, dim=2)
                    edge_pred = remove_extra_class(edge_pred)
                    edge_pred = remove_repeat(edge_pred)
                    edge_pred = add_lost(edge_pred)
                    node_pred = torch.argmax(stroke_emb, dim=1)
                    edge_label = lg2symlg(edge_label, componets)
                except:
                    print(path)
                    edge_label = edge_label.reshape(-1)
                    edge_pred = torch.argmax(edge_emb, dim=1)
                    # all_edge_label = torch.concat((all_edge_label, edge_label.reshape(-1)))
                    # all_edge_pred = torch.concat((all_edge_label, edge_pred.reshape(-1)))
                edge_pred, edge_label = edge_filter(edge_pred,edge_label, los)
                if edge_label.reshape(-1).shape == edge_pred.reshape(-1).shape:
                    all_edge_label = torch.concat((all_edge_label, edge_label.reshape(-1)))
                    all_edge_pred = torch.concat((all_edge_pred, edge_pred.reshape(-1)))




                # node_pred = sym2stroke(node_pred, componets, mask.shape[1])
                # stroke_label = stroke2sym(stroke_label, componets)

                # try:
                #     edge_label = lg2symlg(edge_label, componets)
                #     all_edge_label = torch.concat((all_edge_label, edge_label.reshape(-1)))
                #     # node_pred = torch.argmax(node_out, dim=1)


                # except:
                #     print(path)
                    
                #     all_edge_label = torch.concat((all_edge_label, edge_label.reshape(-1)))
                #     # node_pred = torch.argmax(stroke_emb, dim=1)
                #     error += 1
                #     edge_pred = 
                #     all_edge_pred = torch.concat((all_edge_label, edge_pred.reshape(-1)))
                mask_pred = torch.where(edge_pred == 0, 0, 1)
                mask_label = torch.where(edge_label == 0, 0, 1)

                if torch.equal(node_pred, stroke_label) and torch.equal(edge_pred, edge_label):
                    all_correct += 1
                if torch.equal(edge_pred, edge_label):
                    structure_correct += 1
                if torch.equal(node_pred, stroke_label):
                    edge_correct += 1
                
                all_node_label = torch.concat((all_node_label, stroke_label))
                all_node_pred = torch.concat((all_node_pred, node_pred))

    print(accuracy_score(all_edge_label, all_edge_pred))
    print(precision_score(all_edge_label, all_edge_pred, average='weighted'))
                # else:
                    #  print(edge_pred, edge_label)

    print(all_correct)
    print(edge_correct)
    print(structure_correct)
    print(error)
                # print(edge_pred)
                # print(edge_label)
                # print(torch.sum(edge_pred == edge_label).item())
                # print(torch.sum(edge_pred == edge_label).item()/edge_pred.shape[0]**2)
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_label != 0).item())
                # print(torch.sum(edge_pred == edge_label).item()/torch.sum(edge_pred !=
