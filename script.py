from Model.EdgeGat import EdgeGraphAttention
import torch
edge_gat = EdgeGraphAttention(12, 12,12,12,1)
node = torch.rand(3,4,12)
edge = torch.rand(3,4,4,12)
los = torch.rand(3,4,4)
node_out, edge_out = edge_gat(node, edge, los)