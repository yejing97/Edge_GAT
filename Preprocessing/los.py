import torch
import numpy as np
import math
import intervals as I
PI = math.pi

def LOS(strokes) :
    size = len(strokes)
    edges = np.zeros((size, size))
    for i in range(len(strokes)) : 
        sc = calculate_center(strokes[i])
        U = I.closed(0, 2*PI) # intervals
        # s = sorted(strokes, key = lambda x:distance(x,sc))
        index = sorted(range(len(strokes)), key = lambda x:distance(strokes[x],sc))
        for j in index :
            if i != j: 
                min_theta = math.inf
                max_theta = -math.inf
                for n in strokes[j] :
                    n = np.array(n)
                    w = n - sc
                    h = np.array([1, 0])
                    if w[1] >= sc[1] :
                        theta = math.acos(np.dot(w, h) / np.linalg.norm(w,ord=1) * np.linalg.norm(h,ord=1))
                    else :
                        theta = 2 * PI - math.acos(np.dot(w, h) / np.linalg.norm(w,ord=1) * np.linalg.norm(h,ord=1))
                    min_theta = min(min_theta, theta)
                    max_theta = max(max_theta, theta)
                h = I.closed(min_theta, max_theta)
                # print(h)
                V = U.intersection(h)
                # print(i, j, V)
                if not V.is_empty() :
                    edges[i][j] = 1
                    edges[j][i] = 1
                # updates intervals
                    U = U - h

    return edges

def calculate_center(points) :
    left = math.inf
    right = -math.inf 
    top = -math.inf 
    bottom = math.inf 
    for p in points :
        left = min(left, p[0])
        right = max(right, p[0])
        top = max(top, p[1])
        bottom = min(bottom, p[1])
    return np.array([(left + right) / 2, (top + bottom) / 2])

def distance(s, center) :
    return math.sqrt(pow(center[0] - calculate_center(s)[0], 2) + pow(center[1] - calculate_center(s)[1], 2))