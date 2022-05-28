import json
import sys

from treesegmentation.treeseg_lib import *


def run_treesegmentation(initial_context):
    algorithm = Pipeline(verbose=True) \
        .then(handle_create_file_names_and_paths) \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_gaussian_filter) \
        .then(handle_grid_height_cutoff) \
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

    return algorithm.execute(initial_context)


def load_context_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Context file must be a JSON dictionary.")
    return data


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        settings = load_context_data(args[0])
        run_treesegmentation(settings)
    else:
        print("ts_cli expects a path to a context file path as its only argument.")


if __name__ == "__main__":
    main()
