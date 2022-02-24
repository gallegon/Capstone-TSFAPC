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


def read_las_data(file_path):
    data = laspy.read(file_path)
    header = data.header
    point_count = header.point_count
    scale_xyz = header.scales
    # Offsets currently not being used...
    # offset_xyz = header.offsets
    # Keeping the mins and maxs unscaled, making them ints and not floats.
    # Doing this because laspy sample_data.X, .Y, and .Z are all unscaled ints.
    # However *_min and *_max are all scaled.
    min_xyz = (np.array([header.x_min, header.y_min, header.z_min]) / scale_xyz).astype("int")
    max_xyz = (np.array([header.x_max, header.y_max, header.z_max]) / scale_xyz).astype("int")
    bounds_xyz = (min_xyz, max_xyz)
    range_xyz = (max_xyz - min_xyz) * scale_xyz
    points_xyz = np.array([data.X, data.Y, data.Z])
    return points_xyz, point_count, bounds_xyz, range_xyz, scale_xyz


def run_las2img(points_xyz, bounds_xyz, range_xyz, scale_xyz, discretization, resolution):
    grid_size = np.ceil(range_xyz[:2] / resolution).astype("int")
    cell_size = np.ceil(resolution / scale_xyz[:2]).astype("int")
    grid = las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization)
    return grid, grid_size, cell_size


def run_compute_patches(grid, discretization, min_height, neighbor_mask):
    all_patches = compute_patches(grid, discretization, min_height, neighbor_mask)
    labeled_grid = create_labeled_grid(grid, all_patches)
    # labeled_grid = gaussian_filter(labeled_grid, sigma=0.4)
    compute_patch_neighbors(grid, labeled_grid, all_patches)
    return all_patches, labeled_grid


def run_compute_hierarchies(all_patches):
    return compute_hierarchies(all_patches)


def run_find_connected_hierarchies(contact, hierarchies, weights):
    connected_hierarchies = find_connected_hierarchies(contact)
    hdag = calculate_edge_weight(hierarchies, connected_hierarchies, weights)
    return connected_hierarchies, hdag


def run_partition_graph(hdag, weight_threshold):
    return partition_graph(hdag, weight_threshold)


def run_partitions_to_labeled_grid(partitioned_graph, grid_size):
    labeled_partitions = partitions_to_labeled_grid(partitioned_graph, grid_size[0], grid_size[1])
    # labeled_partitions = gaussian_filter(labeled_partitions, sigma=1)
    return labeled_partitions


def runAlgo(userData):
    print("Radio: ", userData.should_save)
    print("File Path: ", userData.file_path)
    print("Save Path: ", userData.save_path)

    file_path = userData.file_path
    points_xyz, point_count, bounds_xyz, range_xyz, scale_xyz = read_las_data(file_path)
    print(f"== Input data")
    print(f"File: {file_path}")
    print(f"Point count: {point_count}")
    print(f"Range X: {range_xyz[0]:.2f} scaled units")
    print(f"Range Y: {range_xyz[1]:.2f} scaled units")
    print(f"Range Z: {range_xyz[2]:.2f} scaled units")
    print()

    discretization, resolution, min_height = userData.discretization, userData.resolution, userData.min_height
    grid, grid_size, cell_size = run_las2img(points_xyz, bounds_xyz, discretization, resolution)
    print(f"== Creating grid")
    print(f"Resolution: {resolution:.3f} scaled units^2 per pixel")
    print(f"Height levels: {discretization} levels")
    print(f"Height cutoff: {min_height}")
    print(f"Grid dimensions: {grid_size[0]}x{grid_size[1]} cells^2")
    print(f"Cell dimensions: {cell_size[0] * scale_xyz[0]}x{cell_size[1] * scale_xyz[1]} scaled units^2 per cell")
    print()

    all_patches, labeled_grid = run_compute_patches(grid, discretization, min_height, NEIGHBOR_MASK_FOUR_WAY)
    print(f"== Creating patches")
    print(f"Created {len(all_patches)} unique patches")
    print(f"-- Labeling grid")
    print(f"-- Computing cell neighbors")
    print()

    hierarchies, contact = run_compute_hierarchies(all_patches)
    print(f"== Creating hierarchies")
    print(f"Created {len(hierarchies)} unique hierarchies (root nodes)")
    print()

    weights = userData.weights
    connected_hierarchies, hdag = run_find_connected_hierarchies(contact, hierarchies, weights)
    print(f"== Calculating edge weights")
    print(f"Number of unique connected hierarchy pairs: {len(connected_hierarchies)}")
    print()

    weight_threshold = userData.weight_threshold
    partitioned_graph = run_partition_graph(hdag, weight_threshold)
    # partitioned_graph = partition_graph(hdag, userData.weight_threshold)
    print(f"== Partitioning graph (threshold: {weight_threshold})")
    print(f"Number of parentless source nodes: {len(partitioned_graph)}")
    print()

    labeled_partitions = run_partitions_to_labeled_grid(partitioned_graph, grid_size)
    print(f"== Labeling partitions")

    if userData.should_save:
        print(f"Saving to \"{userData.save_path}\"")

        image_grayscale = ((1 - grid / userData.discretization) * 255).transpose().astype("uint8")
        r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()

        for x, y in np.ndindex(len(grid[0]), len(grid[1])):
            # if labeled_partitions[x][y] != 0:
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
