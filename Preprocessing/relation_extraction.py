import numpy as np

def minimal_distance(stroke1, stroke2):
    min_dist = float('inf')  # Initialize with infinity as the minimum distance

    # Calculate the minimum distance between points in stroke1 and stroke2
    for point1 in stroke1:
        for point2 in stroke2:
            dist = np.linalg.norm(np.array(point1) - np.array(point2))  # Euclidean distance
            if dist < min_dist:
                min_dist = dist
    return min_dist

def centroid(points):
    # Calculate centroid (mean point) of a sequence of points
    x_coords, y_coords = zip(*points)
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return centroid_x, centroid_y

def centroid_distances(stroke1, stroke2):
    # Calculate centroids of stroke sequences
    centroid_xi = centroid(stroke1)
    centroid_xj = centroid(stroke2)

    # Calculate horizontal and vertical distances between centroids
    horiz_dist = abs(centroid_xi[0] - centroid_xj[0])  # Horizontal distance
    vert_dist = abs(centroid_xi[1] - centroid_xj[1])   # Vertical distance

    return horiz_dist, vert_dist

