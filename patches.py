# patches.py

import numpy as np
from scipy.ndimage import label

from las2img import las2img


NEIGHBOR_MASK_FOUR_WAY = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
]
NEIGHBOR_MASK_EIGHT_WAY = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]


class Patch:
    ID = 1
    
    def __init__(self, height_level):
        self.height_level = height_level
        self.cells = []
        self.id = Patch.ID
        Patch.ID += 1
    
    def __str__(self):
        lines = [
            f"Patches:",
            f"  height = {self.height_level}",
            f"  cell_count = {len(self.cells)}",
            f"  cells = [",
        ]
        lines.extend(",\n".join([f"    ({x}, {y})" for x, y in self.cells]).splitlines())
        lines.append("  ]")
        return "\n".join(lines)
    
    def __repr__(self):
        return f"Patch<{self.height_level}, {len(self.cells)}, #{self.id}>"


def compute_patches(grid, discretization, min_height, neighbor_mask=None):
    all_patches = []
    for height_level in range(min_height, discretization + 1):
        patches_of_height = find_patches_of_height(grid, height_level, neighbor_mask)
        all_patches.extend(patches_of_height)
    
    return all_patches


def find_patches_of_height(grid, height_level, neighbor_mask=None):
    if neighbor_mask is None:
        neighbor_mask = NEIGHBOR_MASK_FOUR_WAY
    # Create a 2d boolean array of cells (a mask).
    # Cells with height == height_level are True, otherwise False.
    objects_mask = grid == height_level
    # Create a 2d int array of labeled patches from the objects_mask and neighbor_mask.
    # The neighbor_mask dictates which cells are adjacent to each other.
    # Gives each patch a unique label.
    labels, num_labels = label(objects_mask, neighbor_mask)
    
    # For each labeled patch, create a new Patch object containing all the cells in that patch.
    patches_at_height = []
    for i in range(num_labels):
        patch = Patch(height_level)
        cells_xy = np.argwhere(labels == i + 1)
        patch.cells = cells_xy
        patches_at_height.append(patch)
    
    return patches_at_height


def create_labeled_grid(grid, all_patches):
    labeled_grid = np.zeros(grid.shape)
    for patch in all_patches:
        for x, y in patch.cells:
            labeled_grid[x, y] = patch.id
    
    return labeled_grid


def create_and_save_colored_labeled_grid(labeled_grid, all_patches, input_file_path, resolution, discretization):
    from PIL import Image
    import os.path
    
    print("Saving image...");
    # Format the data into an image appropriate format for PIL.
    # Dear god this part needs to be written literally any other way.
    import random
    labeled_raster = labeled_grid.transpose()
    image_grayscale = labeled_raster.astype("uint8")
    r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()
    rmap = {p.id: random.randint(0, 200) for p in all_patches}
    gmap = {p.id: random.randint(0, 200) for p in all_patches}
    bmap = {p.id: random.randint(0, 200) for p in all_patches}
    for i in range(labeled_grid.shape[1]):
        for j in range(labeled_grid.shape[0]):
            id = labeled_raster[i, j]
            if id not in rmap or id == 0:
                r[i, j] = 255
                g[i, j] = 255
                b[i, j] = 255
            else:
                r[i, j] = rmap[id]
                g[i, j] = gmap[id]
                b[i, j] = bmap[id]
    
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")
    # Format the name and save the image.
    save_path = "./labeled_rasters/"
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(save_path, save_name)
    img.save(full_path)
    print(f"Saved image to \"{full_path}\"")


def from_cli():
    import sys
    import os
    from timeit import default_timer as timer
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

    # TODO: Make this a CLI argument.
    min_height_cutoff = 1

    # Run and report the execution time.
    start = timer()
    grid = las2img(file_path, resolution, discretization)
    elapsed = timer() - start
    print(f"Created discretized grid in {elapsed} seconds.")
    start = timer()
    all_patches = compute_patches(grid, discretization, min_height_cutoff, NEIGHBOR_MASK_FOUR_WAY)
    elapsed = timer() - start
    print(f"Computed patches in {elapsed} seconds.")
    labeled_grid = create_labeled_grid(grid, all_patches)
    create_and_save_colored_labeled_grid(labeled_grid, all_patches, file_path, resolution, discretization)


if __name__ == "__main__":
    from_cli()
