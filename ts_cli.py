# ts_cli.py
import random

import laspy
import numpy as np

from scipy.ndimage import gaussian_filter

from treesegmentation import patch, hierarchy, las2img, hdag

if __name__ == "__main__":
    file_path = "sample_data/004_las.las"
    resolution = 1
    discretization = 256
    min_height = 16

    # For weighted graph
    level_depth_weight = 0.84
    node_depth_weight = 1.07
    shared_ratio_weight = -0.11
    top_distance_weight = 0.77
    centroid_distance_weight = 0.0

    # Convert the weights into a numpy array
    weights = np.array([level_depth_weight, node_depth_weight,
                        shared_ratio_weight, top_distance_weight,
                        centroid_distance_weight], dtype=np.float32)

    # graph partitioning threshold
    weight_threshold = 0.4

    print(f"== Input data")
    data = laspy.read(file_path)
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
    print(f"File: {file_path}")
    print(f"Point count: {point_count}")
    print(f"Range X: {range_xyz[0]:.2f} scaled units")
    print(f"Range Y: {range_xyz[1]:.2f} scaled units")
    print(f"Range Z: {range_xyz[2]:.2f} scaled units")

    print()
    print(f"== Creating grid")
    points_xyz = np.array([data.X, data.Y, data.Z])
    grid_size = np.ceil(range_xyz[:2] / resolution).astype("int")
    cell_size = np.ceil(resolution / scale_xyz[:2]).astype("int")
    print(f"Resolution: {resolution:.3f} scaled units^2 per pixel")
    print(f"Height levels: {discretization} levels")
    print(f"Height cutoff: {min_height}")
    print(f"Grid dimensions: {grid_size[0]}x{grid_size[1]} cells^2")
    print(f"Cell dimensions: {cell_size[0] * scale_xyz[0]}x{cell_size[1] * scale_xyz[1]} scaled units^2 per cell")
    grid = las2img.las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization)

    print()
    print(f"== Creating patches")
    all_patches = patch.compute_patches(grid, discretization, min_height, patch.NEIGHBOR_MASK_FOUR_WAY)
    print(f"Created {len(all_patches)} unique patches")
    print(f"-- Labeling grid")
    labeled_grid = patch.create_labeled_grid(grid, all_patches)
    gaussian_filter(labeled_grid, sigma=1)
    print(f"-- Computing cell neighbors")
    patch.compute_patch_neighbors(grid, labeled_grid, all_patches)

    print()
    print(f"== Creating hierarchies")
    hierarchies, contact = hierarchy.compute_hierarchies(all_patches)
    print(f"Created {len(hierarchies)} unique hierarchies (root nodes)")
    
    print()
    print(f"== Calculating edge weights")
    connected_hierarchies = hdag.find_connected_hierarchies(contact)
    print(f"Number of unique connected hierarchy pairs: {len(connected_hierarchies)}")
    HDAG = hdag.calculate_edge_weight(hierarchies, connected_hierarchies, weights)
    print()

    print(f"== Partitioning graph (threshold: {weight_threshold})")
    partitioned_graph = hdag.partition_graph(HDAG, weight_threshold)
    print(f"Number of parentless source nodes: {len(partitioned_graph)}")
    print()

    print(f"== Labeling partitions")
    labeled_partitions = hdag.partitions_to_labeled_grid(partitioned_graph, grid_size[0], grid_size[1])

    print("Save labeled hierarchies as raster? [y/n]")
    user_input = input(">>> ")
    should_save = user_input == 'y' or user_input == 'Y'
    if should_save:
        save_path = "./hierarchy_rasters/"
        print(f"Saving to \"{save_path}\"")
        from PIL import Image
        import os.path

        image_grayscale = ((1 - grid / discretization) * 255).transpose().astype("uint8")
        r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()
        '''
        for hierarchy in hierarchies:
            for y, x in hierarchy.root.patch.cells:
                r[x, y] = 255
                g[x, y] = 0
                b[x, y] = 0
        '''
        for x, y in np.ndindex(len(grid[0]), len(grid[1])):
            #if labeled_partitions[x][y] != 0:
            r[x, y] = labeled_partitions[x][y] % 256
            g[x, y] = (labeled_partitions[x][y] * 2) % 256
            b[x, y] = (labeled_partitions[x][y] * 4) % 256
        '''    
        for i in range(0, grid[0]):
            for j in range(0, grid[1]):
                r[x, y] = labeled_grid[i][j] % 256
                g[x, y] = labeled_grid[i][j] % 256
                b[x, y] = labeled_grid[i][j] % 256
                '''
        image_color = np.dstack((r, g, b))
        img = Image.fromarray(image_color, "RGB")
        # Format the name and save the image.
        save_path = "./labeled_hierarchies/"
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.split(file_path)[1]
        save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
        full_path = os.path.join(save_path, save_name)
        img.save(full_path)
        print(f"Saved image to \"{full_path}\"")
