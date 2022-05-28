import numpy as np


def las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization):
    """Converts a collection of points into an aerial-view raster containing the highest points in each pixel.

    :param points_xyz: Collection of (x, y, z) points to consider.
    :param bounds_xyz: Tuple of the min and max (x, y, z) coordinate.
    :param grid_size: Tuple of the grid width and height (in number of pixels).
    :param cell_size: Tuple of the width and height of each cell (in units of meters per pixel).
    :param discretization: Number of "slices" or z-levels to encode in the image. Should be a power of 2.

    :return: A 2D array containing the highest z value of every point in each cell.
    """

    point_count = points_xyz.shape[1]
    min_xyz, max_xyz = bounds_xyz
    grid_width, grid_height = grid_size
    cell_size_x, cell_size_y = cell_size

    cell_xy_indices = (
            (points_xyz[:2, :] - min_xyz[:2].reshape((2, 1)))
            // np.array([[cell_size_x], [cell_size_y]])
        ).astype("int")

    grid_max_z = np.full(shape=(grid_width, grid_height), fill_value=0)

    def replace_grid_where_z_gt(index, x, y):
        z = points_xyz[2][index]
        if z > grid_max_z[x, y]:
            grid_max_z[x, y] = z

    points_indices = np.arange(point_count)
    compute_grid_point_max_z = np.vectorize(replace_grid_where_z_gt)
    compute_grid_point_max_z(points_indices, cell_xy_indices[0], cell_xy_indices[1])
    # At this point, grid_point_max_z contains the z value of the highest point in each cell.
    # Create the height discretized version of the grid.
    grid_discretized = (grid_max_z / max_xyz[2] * discretization).round().astype("int")
    return grid_discretized
