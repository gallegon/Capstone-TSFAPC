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

import os.path
import sys
from timeit import default_timer as timer

# laspy for reading in LAS files.
import laspy
import numpy as np
# pillow (PIL) for creating and saving an image.
from PIL import Image


def las2img(file_path, resolution, discretization):
    data = laspy.read(file_path)
    header = data.header
    point_count = header.point_count
    scale_xyz = header.scales
    offset_xyz = header.offsets
    # Keeping the mins and maxs unscaled, making them ints and not floats.
    # Doing this because laspy data.X, .Y, and .Z are all unscaled ints.
    # However *_min and *_max are all scaled.
    min_xyz = (np.array([header.x_min, header.y_min, header.z_min]) / scale_xyz).astype("int")
    max_xyz = (np.array([header.x_max, header.y_max, header.z_max]) / scale_xyz).astype("int")
    range_xyz = (max_xyz - min_xyz) * scale_xyz;

    # I have the *unscaled* xyz position of each point
    # points_xyz = [ x: [point_x...]
    #              , y: [point_y...]
    #              , z: [point_z...]
    #              ]
    points_xyz = np.array([data.X, data.Y, data.Z])

    grid_width, grid_height = np.ceil(range_xyz[:2] / resolution).astype("int")
    print(f"Point count: {point_count}")
    print(f"X range: {range_xyz[0]:8.2f} scaled units")
    print(f"Y range: {range_xyz[1]:8.2f} scaled units")
    print(f"Z range: {range_xyz[2]:8.2f} scaled units")
    print(f"Resolution: {resolution:.2f} scaled unit^2 per pixel")
    print(f"Discretization: {discretization} levels")
    print(f"Grid dimensions: {grid_width}x{grid_height}")

    # I want the highest point within each cell
    # I'm going to store this as indices.
    # grid_point_max_z_index will eventually contain the index
    # for the point with the highest z value in each cell.
    grid_point_max_z = np.full(shape=(grid_width, grid_height), fill_value=0)
    cell_size_x, cell_size_y = resolution / scale_xyz[:2]
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


def from_cli():
    # Parse the command line arguments
    args = sys.argv[1:]
    if len(args) == 2:
        discretization = 256
    elif len(args) == 3:
        discretization = int(args[2])
        if not 1 <= discretization <= 256:
            print("Discretization level must be in the range [1, 256].")
            return None
    else:
        print("Invalid number of arguments.")
        print("Usage: las2img.py file_path resolution [discretization]")
        return None
    file_path = args[0]
    if not os.path.exists(file_path):
        print("Specified input file does not exist.")
        return None
    resolution = float(args[1])

    # Run and report the execution time.
    start = timer()
    grid_discretized = las2img(file_path, resolution, discretization)
    elapsed = timer() - start
    print(f"Finished in {elapsed} seconds.")

    print("Saving image...");
    # Format the data into an image appropriate format for PIL.
    image_grayscale = ((1 - grid_discretized / discretization) * 255).astype("uint8").transpose()
    img = Image.fromarray(image_grayscale, "L")
    # Format the name and save the image.
    save_path = "./rasters/"
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.split(file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(save_path, save_name)
    img.save(full_path)
    print(f"Saved image to \"{full_path}\"")


if __name__ == "__main__":
    from_cli()
