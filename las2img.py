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
    point_count = data.header.point_count
    print(f"Point count: {point_count}")
    
    x_scale = data.header.scales[0]
    y_scale = data.header.scales[1]
    z_scale = data.header.scales[2]
    x_offset = data.header.offsets[0]
    y_offset = data.header.offsets[1]
    z_offset = data.header.offsets[2]
    # Keeping *_min, *_max, and *_range unscaled,
    # meaning they are all ints not floats.
    x_min = int(data.header.x_min / x_scale)
    x_max = int(data.header.x_max / x_scale)
    y_min = int(data.header.y_min / y_scale)
    y_max = int(data.header.y_max / y_scale)
    z_min = int(data.header.z_min / z_scale)
    z_max = int(data.header.z_max / z_scale)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    print(f"X range: {x_range * x_scale} meters")
    print(f"Y range: {y_range * y_scale} meters")
    print(f"Z range: {z_range * z_scale} meters")
    
    # Using ceil to capture the very edges of the data as well.
    # Integer rounding or flooring would leave off some information.
    # Type sanity for dimension calculations:
    # [x_range * scale] = meters
    # [resolution] = meters / pixel
    # meters / (meters / pixel) = meters * (pixels / meter) = pixels
    width_grid = ceil(x_range * x_scale / resolution)
    height_grid = ceil(y_range * y_scale / resolution)
    print(f"Resolution: {resolution} meters per pixel")
    print(f"Grid dimensions: {width_grid}x{height_grid} cells")
    
    # Creating a numpy 3 by point_count 2darray
    # This will store the x, y, and z components
    # Structure after the transpose is [point_index][x, y, z]
    points_xyz = np.array([data.X, data.Y, data.Z]).transpose()
    x_cell_size = resolution / x_scale
    y_cell_size = resolution / y_scale
    grid_xyz = np.zeros(shape=(width_grid, height_grid))
    for y_grid in range(height_grid):
        mask_y_grid = (points_xyz[:, 1] - y_min) // y_cell_size == y_grid
        for x_grid in range(width_grid):
            mask_x_grid = (points_xyz[:, 0] - x_min) // x_cell_size == x_grid
            selected = points_xyz[mask_x_grid & mask_y_grid, 2]
            if selected.size == 0:
                highest = z_min
            else:
                highest = np.amax(selected)
            grid_xyz[x_grid, y_grid] = highest
    
    image_grayscale = (grid_xyz - z_min) / (z_max - z_min)
    image_grayscale = (image_grayscale * 255).astype(np.uint8)
    img = Image.fromarray(image_grayscale, "L")
    img.save("raster.png")

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