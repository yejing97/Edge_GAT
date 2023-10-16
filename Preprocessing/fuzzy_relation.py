# fuzzy landscape
import math
import numpy as np
from normalization import calculate_center
PI = math.pi
def fuzzy_se(op, ua):
    return max(0, 1 -  (2/PI)*np.arccos(op.dot(ua)/np.linalg.norm(op)))

def fuzzy_stroke(stroke, point_o, ua):
    # print(stroke)
    fuzzy_stroke = []
    for point in stroke:
        x_p = point[0]
        y_p = point[1]
        op = np.array([x_p - point_o[0], y_p - point_o[1]])
        v_p = fuzzy_se(op, ua)
        fuzzy_stroke.append(v_p)
    # print(fuzzy_stroke)
    return np.array(fuzzy_stroke).astype(np.float32)

def fuzzy_strokes(strokes, los, nb_norm):
    fuzzy_graph = np.zeros((len(strokes), len(strokes), 4, nb_norm))
    ua_0 = np.array([0, 1])
    ua_1 = np.array([0, -1])
    ua_2 = np.array([1, 0])
    ua_3 = np.array([-1, 0])
    for i in range(los.shape[0]):
        for j in range(los.shape[1]):
            if los[i][j] == 1:
                point_o = calculate_center(strokes[i])
                fuzzy_graph[i, j, 0, :len(strokes[j])] = fuzzy_stroke(strokes[j], point_o, ua_0)
                fuzzy_graph[i, j, 1, :len(strokes[j])] = fuzzy_stroke(strokes[j], point_o, ua_1)
                fuzzy_graph[i, j, 2, :len(strokes[j])] = fuzzy_stroke(strokes[j], point_o, ua_2)
                fuzzy_graph[i, j, 3, :len(strokes[j])] = fuzzy_stroke(strokes[j], point_o, ua_3)
    return fuzzy_graph

def fuzzy_relation(stroke_1, stroke_2, nb_norm):    
    # print(len(stroke_1) , len(stroke_2))
    fuzzy_relation = np.zeros((len(stroke_1), 4, len(stroke_2)))
    ua_0 = np.array([0, 1])
    ua_1 = np.array([0, -1])
    ua_2 = np.array([1, 0])
    ua_3 = np.array([-1, 0])
    for i in range(len(stroke_1)):
        # for j in range(len(stroke_2)):
            point_o = stroke_1[i]
            # print(point_o[0])
            fuzzy_relation[i,0,:] = fuzzy_stroke(stroke_2, point_o, ua_0)
            fuzzy_relation[i,1,:] = fuzzy_stroke(stroke_2, point_o, ua_1)
            fuzzy_relation[i,2,:] = fuzzy_stroke(stroke_2, point_o, ua_2)
            fuzzy_relation[i,3,:] = fuzzy_stroke(stroke_2, point_o, ua_3)
    return np.max(fuzzy_relation, axis = 0)

def fuzzy_relations(strokes, los, nb_norm):
    fuzzy_relations = np.zeros((len(strokes), len(strokes), 4, nb_norm))
    for i in range(los.shape[0]):
        for j in range(los.shape[1]):
            if los[i][j] == 1:
                fuzzy_relations[i, j, :, :len(strokes[j])] = fuzzy_relation(strokes[i], strokes[j], nb_norm)
    # print(fuzzy_relations)
    return fuzzy_relations
