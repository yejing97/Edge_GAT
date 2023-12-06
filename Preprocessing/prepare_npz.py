import sys
import multiprocessing

sys.path.append('/home/e19b516g/yejing/code/Edge_GAT/')
print(sys.path)

from Preprocessing.load import load_gt
import Preprocessing.normalization
from Preprocessing.relation_extraction import feature_extract
import os
import numpy as np

def make_npz(tgt_path, inkml, lg, stroke_emb_nb):
    file_id = inkml.split('/')[-1].split('.')[0]

    # load strokes, labels, los_graph
    strokes, stroke_labels, edge_labels, los_graph = load_gt(inkml, lg)
    edge_nb = np.sum(los_graph == 1)
    node_nb = len(strokes) 
    max_len = max(Preprocessing.normalization.stroke_length(stroke)[-1] for stroke in strokes)
    alpha = max_len/stroke_emb_nb

    # relation feature extraction
    # rel_emb = np.zeros((len(strokes), len(strokes), 20), dtype=np.float32)
    sym_emb = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    for i in range(len(strokes)):
    #     # remove speed for symbol feature
    #     # padded_stroke = np.zeros((stroke_emb_nb, 2), dtype=np.float32)
        stroke = Preprocessing.normalization.Speed_norm_stroke(strokes[i], alpha)
        stroke = Preprocessing.normalization.stroke_keep_shape(np.array(stroke))
        sym_emb[i,:len(stroke), :] = np.array(stroke)
    #     for j in range(len(strokes)):
    #         feature = feature_extract(strokes[i], strokes[j], i, j)
    #         rel_emb[i, j, :] = feature
    if not os.path.exists(os.path.join(tgt_path, 'strokes_emb')):
            os.makedirs(os.path.join(tgt_path, 'strokes_emb'))
            # os.makedirs(os.path.join(tgt_path, 'edges_emb'))
            os.makedirs(os.path.join(tgt_path, 'edge_labels'))
            # os.makedirs(os.path.join(tgt_path, 'stroke_labels'))
            # os.makedirs(os.path.join(tgt_path, 'los'))

    np.save(os.path.join(tgt_path, 'strokes_emb' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), sym_emb)
    # np.save(os.path.join(tgt_path, 'edges_emb' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), rel_emb)
    np.save(os.path.join(tgt_path, 'edge_labels' ,'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id), edge_labels)
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
                    make_npz(os.path.join(tgt_path, tag), inkml, lg, stroke_nb)
                # except:
                #     print('error in ' + inkml)
                #     continue

# def make_data(inkml_path, npz_path, stroke_emb_nb):
#     core_nb = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(core_nb)
#     for tag in ['train', 'val', 'test']:
#         pool.map(make_batch_npz, [(tag, inkml_path, npz_path, stroke_emb_nb, id, core_nb) for id in range(core_nb)])
#     pool.close()
#     pool.join()
    # make_batch_npz('train', inkml_path, npz_path, stroke_emb_nb)
    # make_batch_npz('val', inkml_path, npz_path, stroke_emb_nb)
    # make_batch_npz('test', inkml_path, npz_path, stroke_emb_nb)

if __name__ == '__main__':
    root_path = '/home/e19b516g/yejing/data/data_for_graph/'
    inkml_path = os.path.join(root_path, 'INKML')

    S = [150]
    for s in S:
        npz_name = 'S'+ str(s) + 'geo_feat'
        npz_path = os.path.join(root_path, npz_name)
        make_batch_npz('val', inkml_path, npz_path, s, 0, 1)
        # if not os.path.exists(npz_path):
            # make_data(inkml_path, npz_path, s)
        # core_nb = 1
        # pool = multiprocessing.Pool(core_nb)

        # for id in range(core_nb):
        #     pool.apply_async(make_batch_npz, args=('train', inkml_path, npz_path, s, id, core_nb))
        # pool.close()
        # pool.join()

# make_npz('/home/e19b516g/yejing/data/data_for_graph/INKML/train/CROHME2023_train/form_001_E7.inkml', '/home/e19b516g/yejing/data/data_for_graph/LG/train/CROHME2023_train/form_001_E7.lg', 20)