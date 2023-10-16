import numpy as np
import math
def stroke_length(stroke):
    L = 0
    L_n = [0]
    for i in range(len(stroke)-1):
        L = L + np.sqrt((stroke[i+1][0] - stroke[i][0])**2 + (stroke[i+1][1] - stroke[i][1])**2)
        L_n.append(L)
    return L_n

def Speed_norm_stroke(stroke, alpha):
    new_stroke = []

    # calculate stroke length
    L_n = stroke_length(stroke)
    
    m = int(L_n[-1]/alpha)
    # new_stroke.append(stroke[0])
    j = 1
    for p in range(m - 2):
        while L_n[j] < alpha*p:
            j = j + 1
        c = (alpha*p - L_n[j-1])/(L_n[j] - L_n[j-1])
        new_point = (stroke[j - 1][0] + c*(stroke[j][0] - stroke[j-1][0]), stroke[j-1][1] + c*(stroke[j][1] - stroke[j-1][1]))
        new_stroke.append(new_point)
    new_stroke.append(stroke[-1])
        
    return new_stroke

def stroke_keep_shape(stroke):
    s = stroke.reshape([-1, 2]).astype(np.float32)
    if np.min(s[:, 1]) != np.max(s[:, 1]):
        s[:, 1] = (s[:, 1] - np.min(s[:, 1]))/(np.max(s[:, 1]) - np.min(s[:, 1]))
        if np.min(s[:, 0]) != np.max(s[:, 0]):
            s[:, 0] = (s[:, 0] - np.min(s[:, 0]))/(np.max(s[:, 1]) - np.min(s[:, 1]))
        else:
            s[:, 0] = 0.5
    else:
        s[:, 1] = 0.5
        if np.min(s[:, 0]) != np.max(s[:, 0]):
            s[:, 0] = (s[:, 0] - np.min(s[:, 0]))/(np.max(s[:, 0]) - np.min(s[:, 0]))
        else:
            s[:, 0] = 0.5
    return s

def eq_keep_shape(strokes):
    max_x, max_y, min_x, min_y = get_max_min(strokes)
    # new_strokes = np.zeros((len(strokes), norm_nb, 2), dtype=np.float32)

    new_strokes = []
    for i in range(len(strokes)):
        new_stroke = []
        for point in strokes[i]:
            new_point = ((point[0] - min_x)/(max_y - min_y), (point[1] - min_y)/(max_y - min_y))
            new_stroke.append(new_point)
        # n = np.zeros((norm_nb, 2))
        # n[:len(new_stroke),:] = new_stroke
        # new_strokes[i, :, :] = n
        new_strokes.append(new_stroke)
    return new_strokes

def get_max_min(strokes):
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    for stroke in strokes:
        for point in stroke:
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]
            if point[1] > max_y:
                max_y = point[1]
            if point[1] < min_y:
                min_y = point[1]
    return max_x, max_y, min_x, min_y

def calculate_center(stroke):
    left = math.inf
    right = -math.inf 
    top = -math.inf 
    bottom = math.inf 
    for x, y in stroke:
        left = min(left, x)
        right = max(right, x)
        top = max(top, y)
        bottom = min(bottom, y)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return [center_x, center_y]

def norm_centers(centers):
    range_x = np.max(centers[:, 0]) - np.min(centers[:, 0])
    range_y = np.max(centers[:, 1]) - np.min(centers[:, 1])
    if range_x == 0:
        centers[:, 0] = 0.5
    else:
        centers[:, 0] = (centers[:, 0] - np.min(centers[:, 0])) / range_x
    if range_y == 0:
        centers[:, 1] = 0.5
    else:
        centers[:, 1] = (centers[:, 1] - np.min(centers[:, 1])) / range_y
    return centers
