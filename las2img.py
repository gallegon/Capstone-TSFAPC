# las2img.py
#
# Usage:
#   las2img file_path resolution
#
#   `file_path` is a path to the LAS file.
#   `resolution` is in meters per pixel.
#
# Description:
#   Transforms a LAS file into an aerial-view grayscale image.
#   The lowest points are black, the highest points are white.
#   Points for which there is no collected data are colored black.

import os.path
import sys
from timeit import default_timer as timer

# laspy for reading in LAS files.
import laspy
import numpy as np
# pillow (PIL) for creating and saving an image.
from PIL import Image


def main(file_path, resolution):
    data = laspy.read(file_path)
    header = data.header
    point_count = header.point_count
    scale_xyz = header.scales
    offset_xyz = header.offsets
    # Keeping the mins and maxs unscaled, making them ints and not floats.
    min_xyz = (np.array([header.x_min, header.y_min, header.z_min]) / scale_xyz).astype("int")
    max_xyz = (np.array([header.x_max, header.y_max, header.z_max]) / scale_xyz).astype("int")
    range_xyz = (max_xyz - min_xyz) * scale_xyz
    # I have the xyz position of each point
    # points_xyz = [ x: [point_x...]
    #              , y: [point_y...]
    #              , z: [point_z...]
    #              ]
    points_xyz = np.array([data.X, data.Y, data.Z])

    grid_width, grid_height = np.ceil(range_xyz[:2] / resolution).astype("int")
    print(f"Point count: {point_count}")
    print(f"X range: {range_xyz[0]} meters")
    print(f"Y range: {range_xyz[1]} meters")
    print(f"Z range: {range_xyz[2]} meters")
    print(f"Resolution: {resolution} meters per pixel")
    print(f"Image dimensions: {grid_width}x{grid_height}")

    # I want the highest point within each cell
    # I'm going to store this as indices.
    # grid_point_max_z_index will eventually contain the index
    # for the point with the highest z value in each cell.
    grid_point_max_z = np.full(shape=(grid_width, grid_height), fill_value=-1)
    cell_size_x, cell_size_y = resolution / scale_xyz[:2]
    # I need to find the point with the greatest z value within a cell
    # I will do this by turning the point xy coordinates into cell x coordinates
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

    # Format the data into an image appropriate format for PIL.
    image_grayscale = ((grid_point_max_z + 1) / max_xyz[2] * 255).astype("uint8").transpose()
    img = Image.fromarray(image_grayscale, "L")
    # Format the name and save the image.
    save_path = "./rasters/"
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.split(file_path)[1]
    save_name = f"{file_name}_{resolution}-{grid_width}x{grid_height}.png"
    full_path = os.path.join(save_path, save_name)
    img.save(full_path)
    print(f"Saved image to \"{full_path}\"")


if __name__ == "__main__":
    # Doing some basic CLI stuff.
    argv = sys.argv
    if len(argv) != 3:
        print("Invalid number of arguments.")
        print("Usage: las2img file_path resolution")
        exit(1)
    file_path = argv[1]
    if not os.path.exists(file_path):
        print("Invalid file_path: does not exist.")
        exit(2)
    # Resolution in meters per pixel.
    # A value of 1 means each pixel represents
    # the highest point within a meter^2 cell.
    resolution = float(argv[2])
    # Report the execution time.
    start = timer()
    main(file_path, resolution)
    elapsed = timer() - start
    print(f"Finished in {elapsed} seconds.")
