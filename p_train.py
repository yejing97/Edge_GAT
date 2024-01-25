import torch
import os
from tsai.all import *
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from Model.MainModel import Edge_emb
# import load
# import make_pt
# import normalization
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='allsymble')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

model_args = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'class_nb_edge': 8,
    'class_nb_node': 102,
    'epochs': 16,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'min_delta' : 0.00001,
    'patience' : 20,

}

# data_args = {
#     'inkml_path' : '/home/xie-y/data/for_strokes_inkml/',
#     'lg_path' : '/home/xie-y/data/for_strokes_lg/',
#     'pt_path' : '/home/xie-y/data/pt/',
#     'csv_path' : '/home/xie-y/data/csv/',

#     # 'norm_type' :[normalization.stroke_keep_shape, normalization.stroke_tension],
#     # 'norm_nb': [50,100,200],
#     # 'speed_norm' : [normalization.Speed_norm_stroke, normalization.No_speed_norm_stroke],
#     # 'edge_combination' : [load.edge_combination_concate, load.edge_combination_diff]
#     'norm_type' :[normalization.stroke_keep_shape],
#     'norm_nb': [50],
#     'speed_norm' : [normalization.No_speed_norm_stroke],
#     'edge_combination' : [load.edge_combination_diff]

# }

# model_type = [
#     # (xresnet1d18, {}),
#     # (ResNet, {}),
#     # (GRU_FCN, {'shuffle': False}), 
#     # (InceptionTime, {}), 
#     (XceptionTime, {}), 
#     (TransformerModel, {'d_model': 512, 'n_head':4}),
#     ]

# def prepare_data(data_args):
#     for nb in data_args['norm_nb']:
#         for norm in data_args['norm_type']:
#             for speed in data_args['speed_norm']:
#                 pt_name = str(nb) + '_' + norm.__name__ + '_' + speed.__name__
#                 new_tgt_path = os.path.join(data_args['pt_path'],pt_name)
#                 if os.path.exists(new_tgt_path) == False:
#                     make_pt.make_node_pt(
#                         tgt_path= os.path.join(data_args['pt_path'], pt_name, 'node'),
#                         inkml_path=data_args['inkml_path'],
#                         norm_nb=nb,
#                         norm_type=norm,
#                         speed_type=speed
#                     )
#                         # print('no edge')
#                     for edge_c in data_args['edge_combination']:
#                         make_pt.make_edge_pt(
#                             tgt_path=os.path.join(data_args['pt_path'], pt_name, 'edge'),
#                             inkml_path=data_args['inkml_path'],
#                             lg_path=data_args['lg_path'],
#                             norm_nb=nb,
#                             norm_type=norm,
#                             speed_type=speed,
#                             edge_combination=edge_c
#                         )

# def load_data(pt_path, edge_c):
#     if edge_c == 'node':
#         data_path = os.path.join(pt_path, 'node')
#         y = torch.load(os.path.join(data_path, 'y_node.pt'))
#         X = torch.load(os.path.join(data_path, 'X_node.pt'))
#     else:
#         data_path = os.path.join(pt_path,'edge')
#         y = torch.load(os.path.join(data_path, 'y_' + edge_c.__name__ + '.pt'))
#         X = torch.load(os.path.join(data_path, 'X_' + edge_c.__name__ + '.pt'))
#     return X.type(torch.LongTensor), y.type(torch.LongTensor)

# # get best epoch
# def get_max_acc(values):
#     acc_max = 0
#     epoch = 0
#     for i in range(len(values)):
#         acc = values[i][2]
#         if acc > acc_max:
#             acc_max = acc
#             epoch = i
#     return values[epoch]
class Edge_emb_softmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_emb = Edge_emb([40,384,14],0.2)
        self.softmax= torch.nn.Softmax()
    def forward(self,x):
        x = self.edge_emb(x).reshape(-1,14)
        return self.softmax(x)



def train(model_name, model_params, model_args, pt_path, class_nb):
    data_name = pt_path.split('/')[-1]
    y = torch.load(os.path.join(pt_path, 'train_y.pt')).long()
    X = torch.load(os.path.join(pt_path, 'train_X.pt'))
    datasets_train = TSDatasets(X.float(), y)
    val_y = torch.load(os.path.join(pt_path, 'val_y.pt')).long()
    val_X = torch.load(os.path.join(pt_path, 'val_X.pt'))
    datasets_train = TSDatasets(X.float(), y)
    datasets_val = TSDatasets(val_X.float(), val_y)
    print(X.shape)
    print(y.shape)
    dataloader = TSDataLoaders.from_dsets(datasets_train, datasets_val, bs=model_args['batch_size'], num_workers=0)
    if model_name == 'edge':
        softmax = torch.nn.Softmax(dim=-1)
        model = Edge_emb_softmax().to(model_args['device'])
    else:
        model_name_str = str(model_name).split('.')[-2]
        model = create_model(model_name, dls = dataloader, c_in = X.shape[-2], c_out = class_nb, **model_params).to(model_args['device'])
    # print('------' + str(class_nb) + '------data_name:'+ data_name +'------model_name:'+ model_name_str)
    learn = ts_learner(dataloader, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
    cbs = [
        fastai.callback.tracker.EarlyStoppingCallback(min_delta=model_args['min_delta'], 
        patience=model_args['patience']), 
        fastai.callback.tracker.SaveModelCallback(monitor='accuracy', fname='edge', with_opt=True),
        ]
    learn.fit_one_cycle(model_args['epochs'], model_args['lr'], cbs=cbs)
    torch.save(model.state_dict(), './models/node.pth')


def test(pt_path):
    test_y = torch.load(os.path.join(pt_path, 'test_y.pt')).long()
    test_X = torch.load(os.path.join(pt_path, 'test_X.pt'))

# root_path = '/home/xie-y/data/Edge_GAT/S150_R10/'
root_path = '/home/e19b516g/yejing/data/data_for_graph/S150_R10'
# train(model_name=XceptionTime ,model_params = {}, model_args = model_args, pt_path = os.path.join(root_path, args.type), class_nb=102)
train(model_name='edge' ,model_params = {}, model_args = model_args, pt_path = os.path.join(root_path, 'alledge'), class_nb=14)
