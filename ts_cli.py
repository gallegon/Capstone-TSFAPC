from treesegmentation.treeseg_lib import *


def main():
    weight_level_depth = 0.84
    weight_node_depth = 1.07
    weight_shared_ratio = -0.11
    weight_top_distance = 0.77
    weight_centroid_distance = 0.0

    user_data = {
        "input_file_path": "sample_data/hard_nno.las",
        "output_folder_name": None,
        "save_folder_name": None,

        "weight_level_depth": weight_level_depth,
        "weight_node_depth": weight_node_depth,
        "weight_shared_ratio": weight_shared_ratio,
        "weight_top_distance": weight_top_distance,
        "weight_centroid_distance": weight_centroid_distance,
        "weight_threshold": 0.7,

        "resolution": 0.6,
        "discretization": 64,
        "min_height": 10,
        "neighbor_mask": NEIGHBOR_MASK_FOUR_WAY,
        "gaussian": False,
        "gaussian_sigma": 0.1,
        "espg_string": "EPSG:2027",

        "save_grid_raster": True,
        "grid_raster_save_path": "grid_rasters",
        "save_patches_raster": True,
        "patches_raster_save_path": "patches_rasters",
        "save_centroids_raster": False,
        "centroids_raster_save_path": "centroids_raster",
        "save_partition_raster": True,
        "partition_raster_save_path": "partition_rasters",
        "save_tree_raster": False,
        "tree_raster_save_path": "tree_rasters",
        "point_cloud_save_path": "labeled_point_clouds",
    }

    algorithm = Pipeline(verbose=True) \
        .then(handle_create_file_names_and_paths) \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_gaussian_filter) \
        .then(handle_save_grid_raster) \
        .then(handle_compute_patches) \
        .then(handle_patches_to_dict) \
        .then(handle_compute_patches_labeled_grid) \
        .then(handle_compute_patch_neighbors) \
        .then(handle_save_patches_raster) \
        .then(handle_compute_hierarchies) \
        .then(handle_find_connected_hierarchies) \
        .then(handle_calculate_edge_weight) \
        .then(handle_partition_graph) \
        .then(handle_trees_to_labeled_grid) \
        .then(handle_save_partition_raster) \
        .then(handle_label_points) \
        .then(handle_label_point_cloud) \
        .then(handle_save_context_file)

    if algorithm.verbose:
        algorithm.intersperse(transform_print_runtime)

    result = algorithm.execute(user_data)
    return result


if __name__ == "__main__":
    main()
