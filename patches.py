# patches.py
#
# Description:
#   Transforms a LAS file into an aerial-view grayscale image.
#   The lowest points are black, the highest points are white.
#   Points for which there is no collected data are colored black.

import numpy as np
from scipy.ndimage import label, labeled_comprehension

from las2img import las2img

if __name__ == "__main__":
    file_path = "./data/easy_nno.las"
    resolution = 10
    discretization = 32
    
    grid = las2img(file_path, resolution, discretization)
    print("== Discretized Grid")
    print(grid)
    
    height_query = 24
    print(f"== Query Height")
    print(height_query)
    
    objects = grid == height_query
    neighbor_mask = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    labels, num_labels = label(objects, neighbor_mask)
    print("== Unique Patches")
    print(f"Number of patches = {num_labels}")
    print(labels)
    
    print("== Cell Coordinates for each Patch")
    for i in range(1, num_labels + 1):
        cells_xy = np.argwhere(labels == i)
        print(f"-- Height {height_query}, Patch {i}")
        print(cells_xy)
