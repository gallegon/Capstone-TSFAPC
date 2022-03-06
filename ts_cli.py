from treesegmentation.treeseg_lib import *

def main():
    level_depth_weight = 0.84
    node_depth_weight = 1.07
    shared_ratio_weight = -0.11
    top_distance_weight = 0.77
    centroid_distance_weight = 0.0
    weights = np.array([level_depth_weight, node_depth_weight,
                        shared_ratio_weight, top_distance_weight,
                        centroid_distance_weight], dtype=np.float32)

    user_data = {
        "input_file_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\sample_data\\hard_nno.las",
        "weights": weights,
        "weight_threshold": 0.9,
        "resolution": 0.5,
        "discretization": 16,
        "min_height": 4,
        "neighbor_mask": NEIGHBOR_MASK_FOUR_WAY,
        "save_patches_raster": True,
        "patches_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\patches_rasters",
        "save_labeled_raster": True,
        "labeled_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\labeled_rasters"
    }

    algorithm = Pipeline() \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_compute_patches)
    if user_data["save_patches_raster"]:
        algorithm.then(handle_save_patches_raster)
    algorithm \
        .then(handle_compute_hierarchies) \
        .then(handle_find_connected_hierarchies) \
        .then(handle_partition_graph) \
        .then(handle_partitions_to_labeled_grid)
    # handle_save_labeled_grid_as_image has an if checking for should_save as well,
    # so having both ifs is redundant. Doing this to show that there is a lot of
    # flexibility in how the Pipeline and its components are used.
    if user_data["save_labeled_raster"]:
        algorithm.then(handle_save_labeled_raster)

    result = algorithm.execute(user_data)
    print("== Result")
    print(result.keys())

if __name__ == "__main__":
    main()
