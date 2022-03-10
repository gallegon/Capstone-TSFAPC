from treesegmentation.treeseg_lib import *
from treesegmentation.laslabel import laslabel
from timeit import default_timer


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
        "input_file_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\sample_data\\004_las.las",
        "weights": weights,
        "weight_threshold": 0.8,
        "resolution": 1,
        "discretization": 256,
        "min_height": 64,
        "neighbor_mask": NEIGHBOR_MASK_FOUR_WAY,
        "save_grid_raster": True,
        "grid_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\grid_rasters",
        "save_patches_raster": True,
        "patches_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\patches_rasters",
        "save_centroids_raster": False,
        "centroids_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\centroids_raster",
        "save_partition_raster": True,
        "partition_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\partition_rasters",
        "save_tree_raster": True,
        "tree_raster_save_path": "C:\\Users\\moose\\Desktop\\.school\\capstone\\project\\Capstone-TSFAPC\\tree_rasters"
    }

    algorithm = Pipeline() \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_save_grid_raster) \
        .then(handle_compute_patches) \
        .then(handle_compute_patches_labeled_grid) \
        .then(handle_compute_patch_neighbors) \
        .then(handle_save_patches_raster) \
        .then(handle_compute_hierarchies) \
        .then(handle_save_centroids_raster) \
        .then(handle_find_connected_hierarchies) \
        .then(handle_calculate_edge_weight) \
        .then(handle_partition_graph) \
        .then(handle_partitions_to_labeled_grid) \
        .then(handle_save_partition_raster) \
        .then(handle_label_points) \
        .then(handle_save_tree_raster)

    algorithm.intersperse(print_runtime)

    result = algorithm.execute(user_data)
    print("== Result")
    print(result.keys())


if __name__ == "__main__":
    main()
