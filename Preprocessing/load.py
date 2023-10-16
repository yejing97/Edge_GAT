import torch
import xml.etree.ElementTree as ET
import numpy as np
import math
PI = math.pi

import sys

from LG.lg import Lg
from vocab.vocab import vocab

doc_namespace = "{http://www.w3.org/2003/InkML}"

def load_lg(lg_path, dic):
    lg = Lg(lg_path).segmentGraph()
    edge = lg[3]
    node = lg[0]
    strokes = lg[1]
    obj_dic = {}
    strokes_keys = sorted(strokes, key = lambda x:int(x))
    for key in strokes_keys:
        obj_dic[int(key)] = list(strokes[key].values())[0]
    for i in [k for k,v in dic.items() if v == 'None']:
        obj_dic[int(i)] = 'None'
    dic_index = list(sorted(obj_dic.keys()))

    new_matrix = torch.zeros(len(dic), len(dic))
    for key in edge:
        s0 = key[0]
        s1 = key[1]
        relation = list(edge[key].keys())[0]
        index0 = [k for k,v in obj_dic.items() if v == s0]
        index1 = [k for k,v in obj_dic.items() if v == s1]
        for i in index0:
            for j in index1:
                new_matrix[dic_index.index(int(i)), dic_index.index(int(j))] = vocab.rel2indices([relation])[0]
                new_matrix[dic_index.index(int(j)), dic_index.index(int(i))] = vocab.rel2indices([relation])[0] + 6
    for key in node:
        for i in node[key][0]:
            # print(i)
            # print(node[key][0] - set(i))
            for j in node[key][0]:
                if i != j:
                    new_matrix[dic_index.index(int(i)), dic_index.index(int(j))] = vocab.rel2indices(['INNER'])[0]
    return new_matrix

def load_inkml(file_path):
    strokes = []
    labels = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    last_stroke = []
    dic = {}
    for trace_tag in root.findall(doc_namespace + 'traceGroup'):
        for trace_tag in trace_tag.findall(doc_namespace + 'traceGroup'):
            for annotation in trace_tag.findall(doc_namespace + 'annotation'):
                label = annotation.text
            for traceview in trace_tag.findall(doc_namespace + 'traceView'):
                s_id = traceview.get('traceDataRef')
                dic[s_id] = label
    for trace_tag in root.findall(doc_namespace + 'trace'):
        points = []
        last_point = (0, 0)
        for coord in (trace_tag.text).replace('\n', '').split(','):
            this_point = (float(coord.strip().split(' ')[0]), float(coord.strip().split(' ')[1]))
            if this_point != last_point:
                points.append(this_point)
                last_point = this_point
        for coord in trace_tag.items():
            id = coord[1]
        if last_stroke != points:
            strokes.append(points)
            points_np = np.array(points)
            if np.isnan(points_np).any() == True:
                print(file_path)
            if dic.__contains__(id):
                labels.append(dic[id])
            else:
                labels.append('None')
                dic[id] = 'None'
        last_stroke = points
    return strokes, vocab.words2indices(labels), dic

def load_gt(inkml_path, lg_path):
    strokes, s_labels, dic = load_inkml(inkml_path)
    e_labels = load_lg(lg_path, dic)
    return strokes, s_labels, e_labels

