# las2img.py
#
# Usage:
#   las2img file_path resolution
#
#   `file_path` is a path to the LAS file.
#   `resolution` is in meters per pixel.
#
# Description:
#   Transforms a LAS file into an aerial-view discretized grayscale image.
#   The lowest points are white, the highest points are black.
#   Points for which there is no collected data are colored white.


import numpy as np


def las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization):
    point_count = points_xyz.shape[1]
    min_xyz, max_xyz = bounds_xyz
    grid_width, grid_height = grid_size
    cell_size_x, cell_size_y = cell_size
    # I want the highest point within each cell
    # I'm going to store this as indices.
    # grid_point_max_z_index will eventually contain the index
    # for the point with the highest z value in each cell.
    grid_point_max_z = np.full(shape=(grid_width, grid_height), fill_value=0)
    # I need to find the point with the greatest z value within a cell
    # I will do this by turning the point xy coordinates into cell xy coordinates
    # For each point:
    #       point_cell_x = (point_x - min_x) // cell_size_x
    #       point_cell_y = (point_y - min_y) // cell_size_y
    #
    # points_xy_cell_index is an ndarray (2, point_count)
    # points_xy_cell_index = [ x: [point_index...]
    #                        , y: [point_index...]
    #                        ]
    points_xy_cell_index = (
            (points_xyz[:2, :] - min_xyz[:2].reshape((2, 1)))
            // np.array([[cell_size_x], [cell_size_y]])
        ).astype("int")

    def replace_grid_where_z_gt(index, x, y):
        z = points_xyz[2][index]
        if z > grid_point_max_z[x, y]:
            grid_point_max_z[x, y] = z

    points_indices = np.arange(point_count)
    compute_grid_point_max_z = np.vectorize(replace_grid_where_z_gt)
    compute_grid_point_max_z(points_indices, points_xy_cell_index[0], points_xy_cell_index[1])
    # At this point, grid_point_max_z contains the z value of the highest point in each cell.
    # Create the height discretized version of the grid.
    grid_discretized = (grid_point_max_z / max_xyz[2] * discretization).round().astype("int")
    return grid_discretized
