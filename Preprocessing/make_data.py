from Preprocessing.load import load_gt
import Preprocessing.normalization as normalization
import Preprocessing.los as los
import Preprocessing.fuzzy_relation as fuzzy_relation
import argparse
import os
import numpy as np
# from tempfile import TemporaryFile

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/')
# parser.add_argument('--stroke_emb_nb', type=int, default=100)
# parser.add_argument('--rel_emb_nb', type=int, default=5)
# # parser.add_argument('--tgt_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/npz')
# args = parser.parse_args()
# inkml_path = os.path.join(args.root_path, 'INKML')
# lg_path = os.path.join(args.root_path, 'LG')
# npz_path = os.path.join(args.root_path, 'npz')

def make_npz(tgt_path, inkml, lg, stroke_emb_nb, rel_emb_nb, speed):
    file_id = inkml.split('/')[-1].split('.')[0]
    # inkml = os.path.join(inkml_path, file_id + '.inkml')
    # lg = os.path.join(lg_path, file_id + '.lg')
    strokes, stroke_labels, edge_labels = load_gt(inkml, lg)
    # print(edge_labels)
    max_len = max(normalization.stroke_length(stroke)[-1] for stroke in strokes)
    alpha1 = max_len/stroke_emb_nb
    alpha2 = max_len/rel_emb_nb
    strokes_emb = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    # nospeed = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    strokes_no_speed = []
    los_graph = los.LOS(strokes)
    for i in range(len(strokes)):
        if speed:
            new_stroke_node = normalization.Simple_norm_equation(strokes[i], stroke_emb_nb)
            new_stroke_edge = normalization.Simple_norm_equation(strokes[i], rel_emb_nb)

        else:
            new_stroke_node = normalization.Speed_norm_stroke(strokes[i], alpha1)
            new_stroke_edge = normalization.Speed_norm_stroke(strokes[i], alpha2)
        new_stroke_keep_shape = normalization.stroke_keep_shape(np.array(new_stroke_node))
        strokes_emb[i, :len(new_stroke_keep_shape), :] = new_stroke_keep_shape
        strokes_no_speed.append(new_stroke_edge)
    strokes_emb = np.array(strokes_emb)
    edges_emb = fuzzy_relation.fuzzy_relations(strokes_no_speed, los_graph, rel_emb_nb)
    # outfile = TemporaryFile()
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    np.savez(os.path.join(tgt_path, file_id + '.npz'), strokes_emb=strokes_emb, edges_emb=edges_emb, stroke_labels=stroke_labels, edge_labels=edge_labels, los=los_graph)

def make_batch_npz(tag, root_path, tgt_path, stroke_nb, rel_nb, speed):
    for root, _, files in os.walk(os.path.join(root_path, tag)):
        for file in files:
            if file.endswith('.inkml'):
                inkml = os.path.join(root, file)
                lg = os.path.join(root.replace('INKML', 'LG'), file.replace('.inkml', '.lg'))
                if not os.path.exists(lg):
                    if not os.path.exists(os.path.join(root.replace('INKML', 'LG'))):
                        os.makedirs(os.path.join(root.replace('INKML', 'LG')))
                        os.system('convertCrohmeLg '+ root + ' ' + os.path.join(root.replace('INKML', 'LG')))
                    else:
                        print(lg + ' not exists')
                print('make npz for ' + inkml)
                make_npz(os.path.join(tgt_path, tag), inkml, lg, stroke_nb, rel_nb, speed)

# make_batch_npz('train', '/home/e19b516g/yejing/data/data_for_graph/INKML', '/home/e19b516g/yejing/data/data_for_graph/npz', 100, 5)

def make_data(inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed):
    make_batch_npz('train', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed)
    make_batch_npz('val', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed)
    make_batch_npz('test', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed)