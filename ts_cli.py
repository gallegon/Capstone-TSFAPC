# ts_cli.py

import laspy
import numpy as np

from treesegmentation import patch, hierarchy, las2img

if __name__ == "__main__":
    file_path = "sample_data/hard_nno.las"
    resolution = 1
    discretization = 32
    min_height = 16
    
    print(f"Input file: \"{file_path}\"")
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

    # I have the *unscaled* xyz position of each point
    # points_xyz = [ x: [point_x...]
    #              , y: [point_y...]
    #              , z: [point_z...]
    #              ]
    points_xyz = np.array([data.X, data.Y, data.Z])
    grid_size = np.ceil(range_xyz[:2] / resolution).astype("int")
    cell_size = resolution / scale_xyz[:2]
    
    print(f"== Creating grid")
    grid = las2img.las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization)
    
    print()
    print(f"== Creating patches")
    all_patches = patch.compute_patches(grid, discretization, min_height, patch.NEIGHBOR_MASK_FOUR_WAY)
    print(f"Created {len(all_patches)} unique patches")
    print(f"-- Labeling grid")
    labeled_grid = patch.create_labeled_grid(grid, all_patches)
    print(f"-- Computing cell neighbors")
    patch.compute_patch_neighbors(grid, labeled_grid, all_patches)
    
    print()
    print(f"== Creating hierarchies")
    hierarchies = hierarchy.compute_hierarchies(all_patches)
    print(f"Created {len(hierarchies)} unique hierarchies (root nodes)")
    print("----")
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
        for hierarchy in hierarchies:
            for y, x in hierarchy.root.patch.cells:
                r[x, y] = 255
                g[x, y] = 0
                b[x, y] = 0

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
