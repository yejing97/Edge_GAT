import sys
import multiprocessing
import argparse

sys.path.append('/home/e19b516g/yejing/code/Edge_GAT/')
print(sys.path)

from Preprocessing.load import load_gt
import Preprocessing.normalization
from Preprocessing.relation_extraction import feature_extract
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--list', type=list, default=['edge_labels', 'stroke_labels'])
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--tag', type=str, default='train')

args = parser.parse_args()

def make_npz(tgt_path, inkml, lg, stroke_emb_nb, feat_list):
    file_id = inkml.split('/')[-1].split('.')[0]

    # load strokes, labels, los_graph
    strokes, stroke_labels, edge_labels, los_graph = load_gt(inkml, lg)
    edge_nb = np.sum(los_graph == 1)
    node_nb = len(strokes) 
    max_len = max(Preprocessing.normalization.stroke_length(stroke)[-1] for stroke in strokes)
    alpha = max_len/stroke_emb_nb

    # relation feature extraction
    rel_emb = np.zeros((len(strokes), len(strokes), 20), dtype=np.float32)
    sym_emb = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    for i in range(len(strokes)):
    #     # remove speed for symbol feature
    #     # padded_stroke = np.zeros((stroke_emb_nb, 2), dtype=np.float32)
        stroke = Preprocessing.normalization.Speed_norm_stroke(strokes[i], alpha)
        stroke = Preprocessing.normalization.stroke_keep_shape(np.array(stroke))
        sym_emb[i,:len(stroke), :] = np.array(stroke)
        for j in range(len(strokes)):
            feature = feature_extract(strokes[i], strokes[j], i, j)
            rel_emb[i, j, :] = feature
    variables = {
        'sym_emb': sym_emb,
        'rel_emb': rel_emb,
        'edge_labels': edge_labels,
        'stroke_labels': stroke_labels,
        'los_graph': los_graph
    }
    for feat_name in feat_list:

        if not os.path.exists(os.path.join(tgt_path, feat_name)):
            os.makedirs(os.path.join(tgt_path, feat_name))
        # np.save(os.path.join(tgt_path, feat_name ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), sym_emb)
        np.save(os.path.join(tgt_path, feat_name ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), variables[feat_name])
            # os.makedirs(os.path.join(tgt_path, 'edges_emb'))
            # os.makedirs(os.path.join(tgt_path, 'edge_labels'))
            # os.makedirs(os.path.join(tgt_path, 'stroke_labels'))
            # os.makedirs(os.path.join(tgt_path, 'los'))

    # np.save(os.path.join(tgt_path, 'strokes_emb' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), sym_emb)
    # np.save(os.path.join(tgt_path, 'edges_emb' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), rel_emb)
    # np.save(os.path.join(tgt_path, 'edge_labels' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), edge_labels)
    # np.save(os.path.join(tgt_path, 'stroke_labels' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), stroke_labels)
    # np.save(os.path.join(tgt_path, 'los' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), los_graph)
    # return sym_emb, edge_labels, stroke_labels, los_graph

def make_batch_npz(tag, root_path, tgt_path, stroke_nb, id, core_nb):
    for root, _, files in os.walk(os.path.join(root_path, tag)):
        batch_size = len(files)//core_nb
        if id == core_nb - 1:
            start = batch_size*id
            end = len(files)
        else:
            start = batch_size*id
            end = batch_size*(id+1)
        for file in files[start:end]:
            print('process ' + file)
            print('process ' + str(start) + ' to ' + str(end))
            if file.endswith('.inkml'):
                inkml = os.path.join(root, file)
                lg = os.path.join(root.replace('INKML', 'LG'), file.replace('.inkml', '.lg'))
                if os.path.exists(lg):
                #         if not os.path.exists(os.path.join(root.replace('INKML', 'LG'))):
                #             os.makedirs(os.path.join(root.replace('INKML', 'LG')))
                #             os.system('convertCrohmeLg '+ root + ' ' + os.path.join(root.replace('INKML', 'LG')))
                #         else:
                #             print(lg + ' not exists')
                    print('make npz for ' + inkml)
                # try:
                    make_npz(os.path.join(tgt_path, tag), inkml, lg, stroke_nb, args.list)
                # except:
                #     print('error in ' + inkml)
                #     continue

if __name__ == '__main__':
    root_path = '/home/e19b516g/yejing/data/data_for_graph/'
    inkml_path = os.path.join(root_path, 'INKML')
    S = [150]
    for s in S:
        npz_name = 'S'+ str(s) + '_R10'
        npz_path = os.path.join(root_path, npz_name)
        if args.parallel:
            core_nb = multiprocessing.cpu_count() - 5
            pool = multiprocessing.Pool(core_nb)
            for id in range(core_nb):
                # pool.map(make_batch_npz, [args.tag, inkml_path, npz_path, s, id, core_nb])
                pool.apply_async(make_batch_npz, (args.tag, inkml_path, npz_path, s, id, core_nb))
            pool.close()
            pool.join()
        else:
            make_batch_npz(args.tag, inkml_path, npz_path, s, 0, 1)