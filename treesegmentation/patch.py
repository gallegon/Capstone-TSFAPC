# patch.py

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial import distance


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
    
    def __init__(self, height_level, centroid, cells=[]):
        self.height_level = height_level
        self.cell_count = len(cells)
        self.cells = [] if cells is None else cells
        self.neighboring_patches = set()
        self.hierarchies = set()
        self.nearest_hierarchy_id = None
        self.nearest_hierarchy_distance = np.inf
        self.id = Patch.ID
        self.centroid = centroid
        Patch.ID += 1
    
    def __str__(self):
        lines = [
            f"Patch#{self.id}:",
            f"  height = {self.height_level}",
            f"  cell_count = {len(self.cells)}",
            f"  neighbors = [",
            *",\n".join([f"    {neighbor}" for neighbor in self.neighboring_patches]).splitlines(),
            "  ]"
        ]
        return "\n".join(lines)
    
    def __repr__(self):
        return f"Patch<{self.height_level}, {len(self.cells)}, #{self.id}>"

    def add_hierarchy(self, hierarchy):
        self.hierarchies.add(hierarchy)
        for hierarchy in self.hierarchies:
            centroid_distance = (distance.cdist(np.reshape(self.centroid, (-1, 2)),
                                                np.reshape(hierarchy.height_adjusted_centroid, (-1, 2)),
                                                'euclidean')[0][0])

            # If the current hierarchy is closer, assign the patch to it
            if centroid_distance < self.nearest_hierarchy_distance:
                self.nearest_hierarchy_id = hierarchy.root_id
                self.nearest_hierarchy_distance = centroid_distance


def compute_patch_neighbors(grid, labeled_grid, all_patches):
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    grid_width, grid_height = grid.shape
    for patch in all_patches:
        for x, y in patch.cells:
            for ox, oy in offsets:
                xx = x + ox
                yy = y + oy
                if xx < 0 or xx >= grid_width:
                    continue
                if yy < 0 or yy >= grid_height:
                    continue
                other_label = labeled_grid[xx, yy]
                if patch.id == other_label or other_label == 0:
                    continue
                patch.neighboring_patches.add(other_label)
    

def compute_patches(grid, discretization, min_height, neighbor_mask=None):
    all_patches = []
    for height_level in range(min_height, discretization + 1):
        patches_of_height = find_patches_of_height(grid, height_level, neighbor_mask)
        all_patches.extend(patches_of_height)
    return all_patches


def find_patches_of_height(grid, height_level, neighbor_mask=None):
    from scipy.ndimage import gaussian_filter
    neighbor_mask = NEIGHBOR_MASK_FOUR_WAY if neighbor_mask is None else neighbor_mask
    # Create a 2d boolean array of cells (a mask).
    # Cells with height == height_level are True, otherwise False.
    objects_mask = grid == height_level
    # Create a 2d int array of labeled patches from the objects_mask and neighbor_mask.
    # The neighbor_mask dictates which cells are adjacent to each other.
    # Gives each patch a unique label.
    labels, num_labels = label(objects_mask, neighbor_mask)

    # labels = gaussian_filter(labels, sigma=0.3)


    # Find the centroids for these patches
    index = np.arange(1, num_labels + 1)
    centroids = center_of_mass(objects_mask, labels, index)

    # For each labeled patch, create a new Patch object containing all the cells in that patch.
    return [
        Patch(height_level, np.array(centroids[label_id]), 
            np.argwhere(labels == label_id + 1)) for label_id in range(num_labels)
    ]


# Convert the patch list to a dictionary
def patches_to_dict(all_patches):
    patch_dict = {}

    # Map the patch ID the relevant patch
    for patch in all_patches:
        patch_dict[patch.id] = patch

    return patch_dict


def create_labeled_grid(grid, all_patches):
    labeled_grid = np.zeros(grid.shape, dtype="int")
    for patch in all_patches:
        x, y = patch.cells.T
        labeled_grid[x, y] = patch.id
    
    return labeled_grid


def create_and_save_colored_labeled_grid(labeled_grid, all_patches, input_file_path, resolution, discretization):
    from PIL import Image
    import os.path
    
    print("Saving image...")
    # Format the sample_data into an image appropriate format for PIL.
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