# patches.py

import numpy as np
from scipy.ndimage import label

from las2img import las2img


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


def find_patches_of_height(height_level):
    # print(f"== Query Height")
    # print(height_level)
    
    objects = grid == height_level
    neighbor_mask = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    labels, num_labels = label(objects, neighbor_mask)
    # print("== Unique Patches")
    # print(f"Number of patches = {num_labels}")
    # print(labels)
    
    patches_at_height = []
    for i in range(num_labels):
        patch = Patch(height_level)
        cells_xy = np.argwhere(labels == i + 1)
        for x, y in cells_xy:
            patch.cells.append((x, y))
        patches_at_height.append(patch)
    
    # print(f"== Computed Patches of Height {height_level}")
    # for p in patches.items():
    #     print(p)
    
    return patches_at_height


if __name__ == "__main__":
    file_path = "./data/hard_nno.las"
    resolution = 1
    discretization = 5
    
    grid = las2img(file_path, resolution, discretization)
    print("== Discretized Grid")
    # print(grid)
    
    all_patches_by_height = {i: find_patches_of_height(i) for i in range(discretization + 1)}
    all_patches = []
    for ps in all_patches_by_height.values():
        all_patches.extend(ps)
    
    cell_count = sum([len(p.cells) for ps in all_patches_by_height.values() for p in ps])
    print("== Combined Cell Count from Patches")
    print(cell_count)
    
    import random
    MIN_HEIGHT = 1
    labeled_grid = grid.copy()
    for patch in all_patches:
        if patch.height_level < MIN_HEIGHT:
            id = 0
        else:
            id = patch.id
        for x, y in patch.cells:
            labeled_grid[x, y] = id
    
    print(f"== Labeled Grid")
    # print(labeled_grid)
    patch_count = len(all_patches)

    from PIL import Image
    import os
    import os.path
    
    print("Saving image...");
    # Format the data into an image appropriate format for PIL.
    # Dear god this part needs to be written literally any other way.
    labeled_raster = labeled_grid.transpose()
    image_grayscale = labeled_raster.astype("uint8")
    r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()
    rmap = {p.id: random.randint(0, 200) for p in all_patches}
    gmap = {p.id: random.randint(0, 200) for p in all_patches}
    bmap = {p.id: random.randint(0, 200) for p in all_patches}
    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
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
    file_name = os.path.split(file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(save_path, save_name)
    img.save(full_path)
    print(f"Saved image to \"{full_path}\"")
