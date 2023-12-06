import numpy as np
import math

def dist_points(stroke1, stroke2):
    min_dist = float('inf')  # Initialize with infinity as the minimum distance
    max_dist = 0.0  # Initialize with 0 as the maximum distance
    # Calculate the minimum distance between points in stroke1 and stroke2
    for point1 in stroke1:
        for point2 in stroke2:
            dist = np.linalg.norm(np.array(point1) - np.array(point2))  # Euclidean distance
            if dist <= min_dist:
                min_dist = dist
                p1_min = point1
                p2_min = point2
            if dist >= max_dist:
                max_dist = dist
                p1_max = point1
                p2_max = point2

    dx_min = p2_min[0] - p1_min[0]
    dy_min = p2_min[1] - p1_min[1]
    dx_max = p2_max[0] - p1_max[0]
    dy_max = p2_max[1] - p1_max[1]
    return min_dist, dx_min, dy_min, max_dist, dx_max, dy_max

def bounding_box(stroke):
    min_x = min(point[0] for point in stroke)
    max_x = max(point[0] for point in stroke)
    min_y = min(point[1] for point in stroke)
    max_y = max(point[1] for point in stroke)

    # Bounding box coordinates: (min_x, min_y), (max_x, max_y)
    return (min_x, min_y), (max_x, max_y)

def bounding_box_2_storkes(box1, box2):
    (min_x1, min_y1), (max_x1, max_y1) = box1
    (min_x2, min_y2), (max_x2, max_y2) = box2

    # Calculate bounding box of the two strokes
    min_x = min(min_x1, min_x2)
    max_x = max(max_x1, max_x2)
    min_y = min(min_y1, min_y2)
    max_y = max(max_y1, max_y2)
    width = max_x - min_x
    height = max_y - min_y
    # Bounding box coordinates: (min_x, min_y), (max_x, max_y)
    return width, height

def calculate_bounding_box_overlap(box1, box2):
    # Extract coordinates of minimum and maximum points for each bounding box
    (min_x1, min_y1), (max_x1, max_y1) = box1
    (min_x2, min_y2), (max_x2, max_y2) = box2

    # Calculate the intersection coordinates
    inter_min_x = max(min_x1, min_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_x = min(max_x1, max_x2)
    inter_max_y = min(max_y1, max_y2)

    # Calculate the intersection area
    intersect_width = max(0, inter_max_x - inter_min_x)
    intersect_height = max(0, inter_max_y - inter_min_y)

    return intersect_width, intersect_height

def ratio_of_bounding_box_overlap(box1, box2):
    overlap_width, overlap_height = calculate_bounding_box_overlap(box1, box2)
    max_width, max_height = bounding_box_2_storkes(box1, box2)
    return overlap_width*overlap_height/(max_width*max_height + 1e-8)


def distance_of_bounding_box(box1, box2):
    (min_x1, min_y1), (max_x1, max_y1) = box1
    (min_x2, min_y2), (max_x2, max_y2) = box2
    center1 = ((min_x1 + max_x1)/2, (min_y1 + max_y1)/2)
    center2 = ((min_x2 + max_x2)/2, (min_y2 + max_y2)/2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def ratio_area_of_bounding_box(box1, box2):
    (min_x1, min_y1), (max_x1, max_y1) = box1
    (min_x2, min_y2), (max_x2, max_y2) = box2
    area1 = (max_x1 - min_x1)*(max_y1 - min_y1)
    area2 = (max_x2 - min_x2)*(max_y2 - min_y2)
    return area1/(area2 + 1e-8)

def centroid_of_points(points):
    # Calculate centroid (mean point) of a sequence of points
    x_coords, y_coords = zip(*points)
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return centroid_x, centroid_y

def centroid_of_points_dist(stroke1, stroke2):
    # Calculate centroids of stroke sequences
    centroid_xi = centroid_of_points(stroke1)
    centroid_xj = centroid_of_points(stroke2)

    # Calculate horizontal and vertical distances between centroids
    horiz_dist = abs(centroid_xi[0] - centroid_xj[0])  # Horizontal distance
    vert_dist = abs(centroid_xi[1] - centroid_xj[1])   # Vertical distance

    return horiz_dist, vert_dist

def calculate_stroke_length(stroke):
    length = 0.0
    for i in range(1, len(stroke)):
        # Calculate Euclidean distance between consecutive points
        distance = math.sqrt((stroke[i][0] - stroke[i - 1][0])**2 + (stroke[i][1] - stroke[i - 1][1])**2)
        length += distance

    return length

def speed_of_stroke(stroke):
    # Calculate the speed of a stroke
    length = calculate_stroke_length(stroke)
    duration = len(stroke)
    speed = length/(duration + 1e-8)
    return speed

def distance_between_curves(curve1, curve2):
    curve1 = np.array(curve1)
    curve2 = np.array(curve2)
    min_dist = np.inf  # Initialize minimum distance with infinity
    max_dist = 0.0  # Initialize maximum distance with 0
    # Iterate through points in curve1
    for point1 in curve1:
        # Compute distances between point1 and all points in curve2
        distances = np.linalg.norm(curve2 - point1, axis=1)
        
        # Find the minimum distance for point1 to curve2
        min_dist_point = np.min(distances)
        max_dist_point = np.max(distances)
        
        # Update the minimum distance if a smaller distance is found
        if min_dist_point < min_dist:
            min_dist = min_dist_point
        if max_dist_point > max_dist:
            max_dist = max_dist_point
    
    for point2 in curve2:
        # Compute distances between point2 and all points in curve1
        distances = np.linalg.norm(curve1 - point2, axis=1)
        
        # Find the minimum distance for point2 to curve1
        min_dist_point = np.min(distances)
        max_dist_point = np.max(distances)
        
        # Update the minimum distance if a smaller distance is found
        if min_dist_point < min_dist:
            min_dist = min_dist_point
        if max_dist_point > max_dist:
            max_dist = max_dist_point
    
    return min_dist, max_dist

def feature_extract(stroke1, stroke2, pos1, pos2):
    # Calculate the minimum distance between points in stroke1 and stroke2
    # min_dist, dx, dy = minimal_distance(stroke1, stroke2)
    min_dist, dx_min, dy_min, max_dist, dx_max, dy_max = dist_points(stroke1, stroke2)

    min_dist_curve, max_dist_curve = distance_between_curves(stroke1, stroke2)

    # Calculate bounding boxes of stroke1 and stroke2
    box1 = bounding_box(stroke1)
    box2 = bounding_box(stroke2)

    # Calculate centroid distances
    horiz_dist, vert_dist = centroid_of_points_dist(stroke1, stroke2)

    bb_dist = distance_of_bounding_box(box1, box2)

    bb2_weight, bb2_height = bounding_box_2_storkes(box1, box2)
    bb2_area = bb2_weight*bb2_height
    # Calculate stroke length ratio

    ratio_duration = len(stroke1)/(len(stroke2) +1e-8)

    ratio_length = calculate_stroke_length(stroke1)/(calculate_stroke_length(stroke2) + 1e-8)

    ratio_speed = speed_of_stroke(stroke1)/(speed_of_stroke(stroke2) + 1e-8)

    ratio_area = ratio_area_of_bounding_box(box1, box2)

    ratio_bb_overlap = ratio_of_bounding_box_overlap(box1, box2)

    ratio_bb_width = (box1[1][0] - box1[0][0])/(box2[1][0] - box2[0][0] + 1e-8)

    ratio_bb_height = (box1[1][1] - box1[0][1])/(box2[1][1] - box2[0][1] + 1e-8)

    

    temp_dist = pos2 - pos1

    ratio_spatial_temp_dist = min_dist/(temp_dist + 1e-8)


    features = np.array([min_dist, dx_min, dy_min, max_dist, dx_max, dy_max, min_dist_curve, max_dist_curve, horiz_dist, vert_dist, bb_dist, ratio_duration, ratio_length, ratio_speed, ratio_area, ratio_bb_overlap, ratio_bb_width, ratio_bb_height, temp_dist, ratio_spatial_temp_dist])

    return features

    
