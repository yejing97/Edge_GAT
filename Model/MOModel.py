from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from tsai.all import *
# from labml_nn.graphs.gat import GraphAttentionLayer
from sklearn.metrics import accuracy_score
from collections import OrderedDict

# import sys
from Model.EdgeGat import Readout
from Model.MainModel import MainModel
from Model.focalloss import FocalLoss
from Model.min_norm_solvers import MinNormSolver, gradient_normalizers

class MOModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.model = MainModel(args['node_input_size'], args['edge_input_size'], args['edge_gat_input_size'], args['node_gat_input_size'], args['edge_gat_hidden_size'], args['node_gat_hidden_size'], args['edge_gat_output_size'], args['node_gat_output_size'], args['gat_n_heads'], args['dropout'])
        self.readout_edge = Readout(args['edge_gat_output_size'], args['edge_class_nb'])
        self.readout_node = Readout(args['node_gat_output_size'], args['node_class_nb'])

        self.loss_node = torch.nn.CrossEntropyLoss()
        self.loss_edge = torch.nn.CrossEntropyLoss()
        # self.loss_edge = FocalLoss(gamma=2)

        self.validation_step_outputs = []
        self.lr = args['lr']
        self.patience = args['patience']
        self.min_delta = args['min_delta']
        self.d = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_path = args['results_path']

    def load_batch(self, batch):
        strokes_emb, edges_emb, los, strokes_label, edges_label = batch
        strokes_emb = strokes_emb.squeeze(0)
        edges_emb = edges_emb.squeeze(0)
        strokes_label = strokes_label.squeeze(0).long()
        edges_label = edges_label.squeeze(0).reshape(-1).long()
        los = los.squeeze(0).fill_diagonal_(1).unsqueeze(-1)
        return strokes_emb.to(self.d), edges_emb.to(self.d), los.to(self.d), strokes_label.to(self.d), edges_label.to(self.d)
    
    def edge_filter(self, edges_emb, edges_label, los):
        los = los.squeeze().fill_diagonal_(0)
        los = torch.triu(los)
        indices = torch.nonzero(los.reshape(-1)).squeeze()
        edges_label = edges_label[indices]
        edges_emb = edges_emb.reshape(-1, edges_emb.shape[-1])[indices]
        return edges_emb, edges_label
    
    def node_filter(self, nodes_emb):
        nodes = nodes_emb.reshape(int(math.sqrt(nodes_emb.shape[0])), int(math.sqrt(nodes_emb.shape[0])), -1)
        return torch.diagonal(nodes).transpose(0, 1)
    
    def adaptive_scale(self, batch):
        strokes_emb, edges_emb, los, strokes_label, edges_label = self.load_batch(batch)
        # Scaling the loss functions based on the algorithm choice
        loss_data = {}
        grads = {}
        scale = {}
        # mask = None
        # masks = {}
        # use algo MGDA_UB 
        optimizer = self.optimizers()
        optimizer.zero_grad()
        # First compute representations (z)
        # with torch.no_grad():
        node_volatile = strokes_emb
        edge_volatile = edges_emb
            # images_volatile = Variable(images.data)
        # rep = self.model['rep'](images_volatile)
        node_rep, edge_rep = self.model(node_volatile, edge_volatile, los)
        # As an approximate solution we only need gradients for input
        # rep_variable = Variable(rep.data.clone(), requires_grad=True)
        node_rep_variable = node_rep.clone().detach().requires_grad_(True)
        edge_rep_variable = edge_rep.clone().detach().requires_grad_(True)
        # print(node_rep_variable.device, edge_rep_variable.device)
        # Compute gradients of each loss function wrt z
        optimizer.zero_grad()
        out_edge = self.readout_edge(edge_rep_variable)
        out_edge, edges_label = self.edge_filter(out_edge, edges_label, los)
        loss_edge = self.loss_edge(out_edge, edges_label)
        loss_data['edge'] = loss_edge.item()
        loss_edge.backward()
        grads['edge'] = edge_rep_variable.grad.data.clone().requires_grad_(False).to('cpu')
        # print(grads['edge'].device)
        edge_rep_variable.grad.data.zero_()
        optimizer.zero_grad()
        out_node = self.readout_node(node_rep_variable)
        out_node = self.node_filter(out_node)
        loss_node = self.loss_node(out_node, strokes_label)
        loss_data['node'] = loss_node.item()
        loss_node.backward()
        grads['node'] = node_rep_variable.grad.data.clone().requires_grad_(False).to('cpu')
        # print(grads['node'].device)
        node_rep_variable.grad.data.zero_()
        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, normalization_type='loss+')
        for gr_i in range(len(grads['edge'])):
            grads['edge'][gr_i] = grads['edge'][gr_i] / gn['edge']
        for gr_i in range(len(grads['node'])):
            grads['node'][gr_i] = grads['node'][gr_i] / gn['node']
        
        grads['node'] = grads['node'].repeat(1, grads['node'].shape[0]).reshape(grads['edge'].shape[0], grads['edge'].shape[1])
        # print(grads['edge'].shape, grads['node'].shape)

        
        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in ['edge', 'node']])
        scale['edge'] = float(sol[0])
        scale['node'] = float(sol[1])

        return scale

    
    def training_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = self.load_batch(batch)
        scale = self.adaptive_scale(batch)
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        node_hat = self.readout_node(node_hat)
        edge_hat = self.readout_edge(edge_hat)
        edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
        node_hat = self.node_filter(node_hat)
        loss_edge = self.loss_edge(edge_hat, edges_label)
        loss_node = self.loss_node(node_hat, strokes_label)
        loss = scale['node']*loss_node + scale['edge'] * loss_edge
        self.log('train_loss_node', loss_node, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        strokes_emb, edges_emb, los, strokes_label, edges_label = self.load_batch(batch)
        node_hat, edge_hat = self.model(strokes_emb, edges_emb, los)
        node_hat = self.readout_node(node_hat)
        edge_hat = self.readout_edge(edge_hat)
        edge_hat, edges_label = self.edge_filter(edge_hat, edges_label, los)
        node_hat = self.node_filter(node_hat)
        loss_edge = self.loss_edge(edge_hat, edges_label)
        loss_node = self.loss_node(node_hat, strokes_label)
        # scale = self.adaptive_scale(batch)
        # loss = scale['node']*loss_node + scale['edge'] * loss_edge
        self.validation_step_outputs.append([node_hat, strokes_label, edge_hat, edges_label])
        acc_node = accuracy_score(strokes_label.cpu().numpy(), torch.argmax(node_hat, dim=1).cpu().numpy())
        acc_edge = accuracy_score(edges_label.cpu().numpy(), torch.argmax(edge_hat, dim=1).cpu().numpy())
        self.log("val_loss_node", loss_node, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_edge', loss_edge, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_node', acc_node, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc_edge', acc_edge, on_epoch=True, prog_bar=False, logger=True)
        return acc_node
    
    def on_validation_epoch_end(self) -> None:
        # return super().on_validation_epoch_end()
        all_preds = self.validation_step_outputs
        epoch_id = self.current_epoch
        if epoch_id % 10 == 0:
            torch.save(all_preds, os.path.join(self.results_path, 'epoch_' + str(epoch_id) + '.pt'))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, min_lr=self.min_delta, verbose=True),
            'monitor': 'val_acc_node',
            'frequency': self.trainer.check_val_every_n_epoch
        }
    

