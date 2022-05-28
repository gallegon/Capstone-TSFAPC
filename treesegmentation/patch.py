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
    
    def __init__(self, height_level, centroid, cells=None):
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
        """
        Add a hierarchy to a patch.  Determine if a candidate hierarchy is closest to
        the patch centroid.  If it is, set that hierarchy as the nearest hierarchy.

        :param hierarchy: The hierarchy to be examined
        :return: None
        """
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
    """
    Determine the neighbors of a given labeled patch feature in the 2-D discretized
    grid. These neighbors will be used to build the directed Hierarchy graph in
    the hierarchy building step

    :param grid: The discretized grid
    :param labeled_grid: The grid labeled with patch features
    :param all_patches: A list of all the Patch objects
    :return: None
    """
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
    """
    Compute patches by height level in the discretized grid.  For every height
    level that we want to examine in the interval [min_height, discretization],
    find all the patches in that height level and add them to all_patches array.

    :param grid: The discretized grid
    :param discretization: The number of discrete intervals used to create grid
    :param min_height: The minimum height level that we want to create patches for
    :neighbor_mask: The 3x3 adjacency matrix used to determine 4-way or 8-way adjacency.
    :return: all_patches, a list of patches in the grid
    """
    all_patches = []
    for height_level in range(min_height, discretization + 1):
        patches_of_height = find_patches_of_height(grid, height_level, neighbor_mask)
        all_patches.extend(patches_of_height)
    return all_patches


def find_patches_of_height(grid, height_level, neighbor_mask=None):
    """
    For a given height level, find all features of that match that height level
    and create a patch for each unique feature.  This function is called by
    compute patches as a helper function.

    :param grid: The discretized grid
    :param height_level: The height level we are interested in finding patches for
    :neighbor mask: The 3x3 adjacency matrix used to determine 4-way or 8-way adjacency.
    :return: A list of Patch objects for a given height level (if any exist)
    """
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
    """
    Converts the all_patches array to a dictionary.  This is used in the
    weighted graph and partitioning steps for quick lookup of a Patch.
    The keys are the Patch ID and the values are the Patch objects
    associated with that ID.

    :param all_patches: The list of Patch object created by compute_patches
    :return: patch_dict, a dictionary of patches
    """
    patch_dict = {}

    # Map the patch ID the relevant patch
    for patch in all_patches:
        patch_dict[patch.id] = patch

    return patch_dict


def create_labeled_grid(grid, all_patches):
    """
    Label the grid by unique Patch IDs.  This is used to create a Patch raster
    if specified by the user.

    :param grid: The discretized grid
    :param all_patches: A list of all the Patch objects
    :return: labeled_grid, a grid labeled by unique Patch IDs
    """
    labeled_grid = np.zeros(grid.shape, dtype="int")
    for patch in all_patches:
        x, y = patch.cells.T
        labeled_grid[x, y] = patch.id
    
    return labeled_grid
