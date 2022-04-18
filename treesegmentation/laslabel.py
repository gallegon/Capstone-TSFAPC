import numpy as np


def laslabel(labeled_partitions, points_xyz, point_count, min_xyz, cell_size):
    cell_size_x, cell_size_y = cell_size

    def label_point(x, y):
        cell_x, cell_y = (np.array([x, y]) - min_xyz[:2]) // np.array([cell_size_x, cell_size_y])
        label = labeled_partitions[cell_x, cell_y]
        return label

    label_points = np.vectorize(label_point)
    N = point_count
    labeled_points = label_points(points_xyz[0][:N], points_xyz[1][:N])
    return labeled_points


