# run_algorithm.py

from treesegmentation import patch, hierarchy, las2img

if __name__ == "__main__":
    file_path = "sample_data/easy_nno.las"
    resolution = 1
    discretization = 16
    min_height = 5
    
    print(f"Input file: \"{file_path}\"")
    print(f"== Creating grid")
    grid = las2img.las2img(file_path, resolution, discretization)
    
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
    print("Save all hierarchies as rasters? [y/n]")
    user_input = input(">>> ")
    should_save = user_input == 'y' or user_input == 'Y'
    if should_save:
        save_path = "./hierarchy_rasters/"
        print(f"Saving {len(hierarchies)} image(s) to \"{save_path}\"")
        hierarchy_grids = {h.root_id: hierarchy.hierarchy_as_raster(labeled_grid, h) for h in hierarchies}
        for root_id, grid_of_hierarchy in hierarchy_grids.items():
            from PIL import Image
            import os.path
        
            # Format the sample_data into an image appropriate format for PIL.
            image_grayscale = grid_of_hierarchy.astype("uint8").transpose()
            img = Image.fromarray(image_grayscale, "L")
            # Format the name and save the image.
            os.makedirs(save_path, exist_ok=True)
            file_name = os.path.split(file_path)[1]
            save_name = f"{file_name}_h#{root_id}-{resolution}-{discretization}-{img.width}x{img.height}.png"
            full_path = os.path.join(save_path, save_name)
            img.save(full_path)
