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
from math import ceil
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
    print(f"Point count: {point_count}")
    print(f"X range: {range_xyz[0]} meters")
    print(f"Y range: {range_xyz[1]} meters")
    print(f"Z range: {range_xyz[2]} meters")
    # I have the xyz position of each point
    points_xyz = np.array([data.X, data.Y, data.Z])

    # I want the highest point within each cell
    # I'm going to store this as indices.
    cell_size_x, cell_size_y = resolution / scale_xyz[:2]
    grid_width, grid_height = np.ceil(range_xyz[:2] / resolution).astype("int")
    # print(cell_size_x, cell_size_y)
    # print(grid_width, grid_height)
    grid_indices = np.full(shape=(grid_width, grid_height), fill_value=min_xyz[2])
    # For every point in the point cloud I need to:
    # - Compute it's location in the grid_indices array
    #   - By // cell_size
    # - Update grid_indices to point to the point if it's z is greater than the current.

    # I need to find the point with the greatest z value within a cell
    # I will do this by turning the xy coordinates into grid coordinates
    cells_xy = ((points_xyz[:2, :] - min_xyz[:2].reshape((2, 1)))
                // np.array([[cell_size_x], [cell_size_y]])).astype("int")
    print(cells_xy)
    # Find the tallest point in cell 0, 0
    # mask_in_row_0 = cells_xy[0] == 0
    # mask_in_col_0 = cells_xy[1] == 0
    # mask_in_cell_0_0 = mask_in_row_0 & mask_in_col_0
    # print(mask_in_row_0)
    # print(mask_in_col_0)
    # print(mask_in_cell_0_0)
    points_indices = np.arange(point_count)
    # points_z_in_cell_0_0 = points_xyz[2][mask_in_cell_0_0]
    # indices_in_cell_0_0 = points_indices[mask_in_cell_0_0]
    # print(indices_in_cell_0_0)
    # print(points_z_in_cell_0_0)
    # tallest_index_in_cell_0_0 = indices_in_cell_0_0[points_z_in_cell_0_0.argmax()]
    # print(tallest_index_in_cell_0_0)
    # tallest_point_in_cell_0_0 = points_xyz[:, tallest_index_in_cell_0_0]
    # print(tallest_point_in_cell_0_0)

    # Find the tallest point in every cell
    # Store it in grid_indices[grid_x, grid_y] = tallest_point_index
    # Create a mask for the tallest point in each row
    # Create a mask for the tallest point in each column
    # Output is the grid, where each element is an index to the highest point in that cell
    # For each row, find the tallest point in the row
    points_indices = np.arange(point_count)
    mask_in_row_0 = cells_xy[0] == 0
    # mask_in_cell_0_0 = mask_in_row_0 & mask_in_col_0
    # print(mask_in_row_0)
    # print(mask_in_col_0)
    # print(mask_in_cell_0_0)
    # points_z_in_cell_0_0 = points_xyz[2][mask_in_cell_0_0]
    # indices_in_cell_0_0 = points_indices[mask_in_cell_0_0]
    # print(indices_in_cell_0_0)
    # print(points_z_in_cell_0_0)

    # indices_in_cell_0_0 = points_indices[mask_in_cell_0_0]
    # print(indices_in_cell_0_0)


    print("done")

    # I need to find the point with the greatest z value
    # highest_z = points_xyz[2, :].argmax()
    # print(points_xyz[:, highest_z])


    # x_scale = data.header.scales[0]
    # y_scale = data.header.scales[1]
    # z_scale = data.header.scales[2]
    # x_offset = data.header.offsets[0]
    # y_offset = data.header.offsets[1]
    # z_offset = data.header.offsets[2]
    # # Keeping *_min, *_max, and *_range unscaled,
    # # meaning they are all ints not floats.
    # x_min = int(data.header.x_min / x_scale)
    # x_max = int(data.header.x_max / x_scale)
    # y_min = int(data.header.y_min / y_scale)
    # y_max = int(data.header.y_max / y_scale)
    # z_min = int(data.header.z_min / z_scale)
    # z_max = int(data.header.z_max / z_scale)
    # x_range = x_max - x_min
    # y_range = y_max - y_min
    # z_range = z_max - z_min
    # print(f"X range: {x_range * x_scale} meters")
    # print(f"Y range: {y_range * y_scale} meters")
    # print(f"Z range: {z_range * z_scale} meters")
    
    # Using ceil to capture the very edges of the data as well.
    # Integer rounding or flooring would leave off some information.
    # Type sanity for dimension calculations:
    # [x_range * scale] = meters
    # [resolution] = meters / pixel
    # meters / (meters / pixel) = meters * (pixels / meter) = pixels
    # width_grid = ceil(x_range * x_scale / resolution)
    # height_grid = ceil(y_range * y_scale / resolution)
    # print(f"Resolution: {resolution} meters per pixel")
    # print(f"Grid dimensions: {width_grid}x{height_grid} cells")
    
    # Creating a numpy 3 by point_count 2darray
    # This will store the x, y, and z components
    # Structure after the transpose is [point_index][x, y, z]
    # points_xyz = np.array([data.X, data.Y, data.Z]).transpose()
    # x_cell_size = resolution / x_scale
    # y_cell_size = resolution / y_scale

    
    # image_grayscale = (grid - z_min) / (z_max - z_min)
    # image_grayscale = (image_grayscale * 255).astype(np.uint8)
    # img = Image.fromarray(image_grayscale, "L")
    # # Format the name and save the image.
    # save_path = "./rasters/"
    # os.makedirs(save_path, exist_ok=True)
    # file_name = os.path.split(file_path)[1]
    # save_name = f"{file_name}_{resolution}-{width_grid}x{height_grid}.png"
    # full_path = os.path.join(save_path, save_name)
    # img.save(full_path)
    # print(f"Saved image to \"{full_path}\"")


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
