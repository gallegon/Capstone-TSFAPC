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
# pillow (PIL) for creating and saving an image.
from PIL import Image

def main(file_path, resolution):
    data = laspy.read(file_path)
    print(f"Point count: {data.header.point_count}")
    scale = data.header.scales[0]
    offset = data.header.offsets[0]
    
    # Keeping *_min, *_max, and *_range unscaled,
    # meaning they are all ints not floats.
    x_min = int(data.header.x_min / scale)
    x_max = int(data.header.x_max / scale)
    y_min = int(data.header.y_min / scale)
    y_max = int(data.header.y_max / scale)
    z_min = int(data.header.z_min / scale)
    z_max = int(data.header.z_max / scale)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    print(f"X range: {x_range * scale} meters")
    print(f"Y range: {y_range * scale} meters")
    print(f"Z range: {z_range * scale} meters")
    
    # Using ceil to capture the very edges of the data as well.
    # Integer rounding or flooring would leave off some information.
    # Type sanity for dimension calculations:
    # [x_range * scale] = meters
    # [resolution] = meters / pixel
    # meters / (meters / pixel) = meters * (pixels / meter) = pixels
    img_width = ceil(x_range * scale / resolution)
    img_height = ceil(y_range * scale / resolution)
    print(f"Resolution: {resolution} meters per pixel")
    print(f"Image dimensions: {img_width}x{img_height} pixels^2")
    
    # Create a grid based off the resolution
    # and find the highest point in each cell.
    grid = {}
    cell_size = resolution / scale
    for point in data:
        x, y, z = point.X, point.Y, point.Z
        x_grid = int((x - x_min) // cell_size)
        y_grid = int((y - y_min) // cell_size)
        id_grid = (x_grid, y_grid)
        if not id_grid in grid or grid[id_grid] < z:
            grid[id_grid] = z
    
    # Create a grayscale image and default it to black.
    img = Image.new(mode="L", size=(img_width, img_height))
    for i in range(img_width):
        for j in range(img_height):
            img.putpixel((i,j), 0)
    # For every collected point, put it on the picture.
    for xy, z in grid.items():
        # Normalize z and convert it to a range of [0, 255]
        # (0 is black, 255 is white).
        val = int((z - z_min) / (z_max - z_min) * 255)
        img.putpixel(xy, val)
    
    # Format the name and save the image.
    save_path = "./rasters/"
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.split(file_path)[1]
    save_name = f"{file_name}_{resolution}-{img_width}x{img_height}.bmp"
    full_path = os.path.join(save_path, save_name)
    img.save(full_path, "BMP")
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