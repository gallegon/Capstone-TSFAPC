import laspy
import numpy as np
from .hdag import *
from .hierarchy import *
from .las2img import *
from .patch import *

from PIL import Image
import os.path

class UserInfo:
    def __init__ (self):
        self.file_path = None
        self.resolution = None
        self.discretization = None
        self.min_height = None
        self.should_save = None
        self.save_path = None
        self.weights = None
        self.weight_threshold = None


# def read_lasdata()


# las2img

# compute_patches
# create_labeled_grid
# compute_patch_neighbors

# compute_hierarchies

# find_connected_hierarchies
# calculate_edge_weight

# partition_graph

# partitions_to_labeled_grid

def runAlgo(userData):

    print("Radio: ", userData.should_save)
    print("File Path: ", userData.file_path)
    print("Save Path: ", userData.save_path)

    print(f"== Input data")
    data = laspy.read(userData.file_path)
    header = data.header
    point_count = header.point_count
    scale_xyz = header.scales
    offset_xyz = header.offsets
    # Keeping the mins and maxs unscaled, making them ints and not floats.
    # Doing this because laspy sample_data.X, .Y, and .Z are all unscaled ints.
    # However *_min and *_max are all scaled.
    min_xyz = (np.array([header.x_min, header.y_min, header.z_min]) / scale_xyz).astype("int")
    max_xyz = (np.array([header.x_max, header.y_max, header.z_max]) / scale_xyz).astype("int")
    bounds_xyz = (min_xyz, max_xyz)
    range_xyz = (max_xyz - min_xyz) * scale_xyz
    print(f"File: {userData.file_path}")
    print(f"Point count: {point_count}")
    print(f"Range X: {range_xyz[0]:.2f} scaled units")
    print(f"Range Y: {range_xyz[1]:.2f} scaled units")
    print(f"Range Z: {range_xyz[2]:.2f} scaled units")
    
    print()
    print(f"== Creating grid")
    points_xyz = np.array([data.X, data.Y, data.Z])
    grid_size = np.ceil(range_xyz[:2] / userData.resolution).astype("int")
    cell_size = np.ceil(userData.resolution / scale_xyz[:2]).astype("int")
    print(f"Resolution: {userData.resolution:.3f} scaled units^2 per pixel")
    print(f"Height levels: {userData.discretization} levels")
    print(f"Height cutoff: {userData.min_height}")
    print(f"Grid dimensions: {grid_size[0]}x{grid_size[1]} cells^2")
    print(f"Cell dimensions: {cell_size[0] * scale_xyz[0]}x{cell_size[1] * scale_xyz[1]} scaled units^2 per cell")
    grid = las2img(points_xyz, bounds_xyz, grid_size, cell_size, userData.discretization)
    
    print()
    print(f"== Creating patches")
    all_patches = compute_patches(grid, userData.discretization, userData.min_height, NEIGHBOR_MASK_FOUR_WAY)
    print(f"Created {len(all_patches)} unique patches")
    print(f"-- Labeling grid")
    labeled_grid = create_labeled_grid(grid, all_patches)
    # labeled_grid = gaussian_filter(labeled_grid, sigma=0.4)
    print(f"-- Computing cell neighbors")
    compute_patch_neighbors(grid, labeled_grid, all_patches)
    
    print()
    print(f"== Creating hierarchies")
    hierarchies, contact = compute_hierarchies(all_patches)
    print(f"Created {len(hierarchies)} unique hierarchies (root nodes)")

    print()
    print(f"== Calculating edge weights")
    connected_hierarchies = find_connected_hierarchies(contact)
    print(f"Number of unique connected hierarchy pairs: {len(connected_hierarchies)}")
    HDAG = calculate_edge_weight(hierarchies, connected_hierarchies, userData.weights)
    print()

    print(f"== Partitioning graph (threshold: {userData.weight_threshold})")
    partitioned_graph = partition_graph(HDAG, userData.weight_threshold)
    print(f"Number of parentless source nodes: {len(partitioned_graph)}")
    print()

    print(f"== Labeling partitions")
    labeled_partitions = partitions_to_labeled_grid(partitioned_graph, grid_size[0], grid_size[1])
    # labeled_partitions = gaussian_filter(labeled_partitions, sigma=1)


    
    if userData.should_save:
        print(f"Saving to \"{userData.save_path}\"")


        image_grayscale = ((1 - grid / userData.discretization) * 255).transpose().astype("uint8")
        r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()

        for x, y in np.ndindex(len(grid[0]), len(grid[1])):
            #if labeled_partitions[x][y] != 0:
            r[x, y] = labeled_partitions[x][y] % 256
            g[x, y] = (labeled_partitions[x][y] * 2) % 256
            b[x, y] = (labeled_partitions[x][y] * 4) % 256

        image_color = np.dstack((r, g, b))
        img = Image.fromarray(image_color, "RGB")
        # Format the name and save the image.
        # save_path = "./labeled_hierarchies/"
        os.makedirs(userData.save_path, exist_ok=True)
        file_name = os.path.split(userData.file_path)[1]
        save_name = f"{file_name}_{userData.resolution}-{userData.discretization}-{img.width}x{img.height}.png"
        full_path = os.path.join(userData.save_path, save_name)
        print(f"Saved image to \"{full_path}\"")
        img.save(full_path)
