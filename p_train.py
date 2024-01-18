import torch
import os
from tsai.all import *
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
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
    'epochs': 100,
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

def train(model_name, model_params, model_args, pt_path, class_nb):
    # if edge_c == 'node':
    data_name = pt_path.split('/')[-1]
    # csv_path = data_args['csv_path']
    # X, y = load_data(pt_path, edge_c)
    y = torch.load(os.path.join(pt_path, 'train_y.pt')).long()
    X = torch.load(os.path.join(pt_path, 'train_X.pt'))
    datasets_train = TSDatasets(X.float(), y)
    val_y = torch.load(os.path.join(pt_path, 'val_y.pt')).long()
    val_X = torch.load(os.path.join(pt_path, 'val_X.pt'))
    datasets_train = TSDatasets(X.float(), y)
    datasets_val = TSDatasets(val_X.float(), val_y)

    dataloader = TSDataLoaders.from_dsets(datasets_train, datasets_val, bs=model_args['batch_size'], num_workers=0)
    # for model_name in model_type:
    # for (model_name, model_params) in model_type:
    if model_name == xresnet1d18:
        model_name_str = 'xresnet1d18'
        model = xresnet1d18(c_in = X.shape[-2], c_out = class_nb).to(model_args['device'])
    else:
        model_name_str = str(model_name).split('.')[-2]
        model = create_model(model_name, dls = dataloader, c_in = X.shape[-2], c_out = class_nb, **model_params).to(model_args['device'])
    print('------' + str(class_nb) + '------data_name:'+ data_name +'------model_name:'+ model_name_str)
    learn = ts_learner(dataloader, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
    start = time.time()
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_acc_node',
    #     mode='max',
    #     dirpath='./checkpoints',
    #     filename= 'pretrain' + '-{epoch:02d}-{val_acc_node:.2f}',
    #     save_top_k=3,
    #     save_last=True,
    # )
    cbs = [
        fastai.callback.tracker.EarlyStoppingCallback(min_delta=model_args['min_delta'], 
        patience=model_args['patience']), 
        fastai.callback.tracker.SaveModelCallback(monitor='accuracy', fname=model_name_str, with_opt=True),
        ]
    learn.fit_one_cycle(model_args['epochs'], model_args['lr'], cbs=cbs)
    elapsed = time.time() - start
    # vals = get_max_acc(learn.recorder.values)
    # results.loc[i] = [data_name, model_name_str, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
    # i = i + 1
    # values_df = pd.DataFrame(learn.recorder.values, columns=['train_loss', 'valid_loss', 'accuracy'])
    # values_df.to_csv(os.path.join(csv_path, str(data_namee) + '_' + model_name_str + '.csv'))


# def benchmark(model_type, model_args, data_args):
#     # prepare_data(data_args)
#     results_node = pd.DataFrame(columns=['data_type', 'model_type', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
#     results_edge = pd.DataFrame(columns=['data_type', 'model_type', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
#     i = 0
#     j = 0
#     for nb in data_args['norm_nb']:
#         for norm in data_args['norm_type']:
#             for speed in data_args['speed_norm']:
#                 pt_name = str(nb) + '_' + norm.__name__ + '_' + speed.__name__
#                 new_tgt_path = os.path.join(data_args['pt_path'],pt_name)
#                 # try:
#                 for (model_name, model_params) in model_type:
#                     train(model_name, model_params, model_args, new_tgt_path, model_args['class_nb_node'], 'node', results_node, i)
#                     i = i + 1
#                     for edge_c in data_args['edge_combination']:
#                         train(model_name, model_params, model_args, new_tgt_path, model_args         ['class_nb_edge'],edge_c , results_edge, j)
#                         j = j + 1
#                 # except:
#                 #     print('error:' + pt_name)
#                 #     continue
#     try:
#         results_node.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
#         results_node.to_csv(os.path.join(data_args['csv_path'], ' node_results.csv'))
#         results_edge.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
#         results_edge.to_csv(os.path.join(data_args['csv_path'], ' edge_results.csv'))
#     except:
#         print('csv error')




# prepare_data(data_args)
# benchmark(model_type, model_args, data_args)
root_path = '/home/xie-y/data/Edge_GAT/S150_R10/'
# root_path = '/home/e19b516g/yejing/data/data_for_graph/S150_R10'
train(model_name=XceptionTime ,model_params = {}, model_args = model_args, pt_path = os.path.join(root_path, args.type), class_nb=102)