import sys
import os

from Preprocessing.load import load_gt
import Preprocessing.normalization as normalization
import Preprocessing.los as los
import Preprocessing.fuzzy_relation as fuzzy_relation
import argparse
import numpy as np

# from tempfile import TemporaryFile

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/')
parser.add_argument('--stroke_emb_nb', type=int, default=100)
parser.add_argument('--rel_emb_nb', type=int, default=5)
# parser.add_argument('--tgt_path', type=str, default='/home/e19b516g/yejing/data/data_for_graph/npz')
args = parser.parse_args()
inkml_path = os.path.join(args.root_path, 'INKML')
# lg_path = os.path.join(args.root_path, 'LG')
# npz_path = os.path.join(args.root_path, 'npz')

def make_npz(tgt_path, inkml, lg, stroke_emb_nb, rel_emb_nb, speed, norm):
    file_id = inkml.split('/')[-1].split('.')[0]
    # inkml = os.path.join(inkml_path, file_id + '.inkml')
    # lg = os.path.join(lg_path, file_id + '.lg')
    strokes, stroke_labels, edge_labels, los_graph = load_gt(inkml, lg)
    # print(edge_labels)
    max_len = max(normalization.stroke_length(stroke)[-1] for stroke in strokes)
    alpha1 = max_len/stroke_emb_nb
    alpha2 = max_len/rel_emb_nb
    strokes_emb = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    # nospeed = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
    strokes_no_speed_edge = []
    strokes_no_speed_node = []
    # los_graph = los.LOS(strokes)
    edge_nb = np.sum(los_graph == 1)
    node_nb = len(strokes)
    for i in range(len(strokes)):
        if speed:
            new_stroke_node = normalization.Simple_norm_equation(strokes[i], stroke_emb_nb)
            new_stroke_edge = normalization.Simple_norm_equation(strokes[i], rel_emb_nb)

        else:
            new_stroke_node = normalization.Speed_norm_stroke(strokes[i], alpha1)
            new_stroke_edge = normalization.Speed_norm_stroke(strokes[i], alpha2)
        new_stroke_keep_shape = normalization.stroke_keep_shape(np.array(new_stroke_node))
        strokes_emb[i, :len(new_stroke_keep_shape), :] = new_stroke_keep_shape
        strokes_no_speed_edge.append(new_stroke_edge)
        strokes_no_speed_node.append(new_stroke_node)
    if norm == 'stroke':
        strokes_emb = np.array(strokes_emb)
    elif norm == 'equation':
        new_strokes_keep_shape = normalization.eq_keep_shape(strokes_no_speed_node)
        strokes_emb = np.zeros((len(strokes), stroke_emb_nb, 2), dtype=np.float32)
        for i in range(len(new_strokes_keep_shape)):
            strokes_emb[i, :len(new_strokes_keep_shape[i]), :] = new_strokes_keep_shape[i]
        strokes_emb = np.array(strokes_emb)
    edges_emb = fuzzy_relation.fuzzy_relations(strokes_no_speed_edge, los_graph, rel_emb_nb)
    # outfile = TemporaryFile()
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    np.savez(os.path.join(tgt_path, 'N'+ str(node_nb) + 'E' + str(edge_nb) + '_' + file_id + '.npz'), strokes_emb=strokes_emb, edges_emb=edges_emb, stroke_labels=stroke_labels, edge_labels=edge_labels, los=los_graph)

def make_batch_npz(tag, root_path, tgt_path, stroke_nb, rel_nb, speed, norm):
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
                try:
                    make_npz(os.path.join(tgt_path, tag), inkml, lg, stroke_nb, rel_nb, speed, norm)
                except:
                    print('error in ' + inkml)
                    continue

# make_batch_npz('train', '/home/e19b516g/yejing/data/data_for_graph/INKML', '/home/e19b516g/yejing/data/data_for_graph/npz', 100, 5)

def make_data(inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed, norm):
    make_batch_npz('train', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed, norm)
    make_batch_npz('val', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed, norm)
    make_batch_npz('test', inkml_path, npz_path, stroke_emb_nb, rel_emb_nb, speed, norm)

if __name__ == '__main__':
    S = [50, 100, 150, 200]
    R = [5, 10]
    for s in S:
        for r in R:
            npz_name = 'S'+ str(s) + '_R' + str(r) + '_stroke'
            npz_path = os.path.join(args.root_path, npz_name)
            if not os.path.exists(npz_path):
                make_data(inkml_path, npz_path, s, r, False, 'stroke')