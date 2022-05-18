import os.path
import timeit
from PIL import Image
import pdal
import json
import time
import numpy as np

from .hdag import *
from .hierarchy import *
from .las2img import *
from .patch import *
from .tree import *
from .laslabel import *


class Pipeline:
    """ Controls the order and handling of multiple stages of a process.
    Build a pipeline by constructing a Pipeline and using the .then method
    to add stages.
    Use the .execute method to pass an initial input and begin running the
    first stage. Each stage is executed sequentially, passing output from
    the previous stage as input to the next stage.
    Either every stage completes successfully, or somewhere along the line
    an error occurs. Therefore the .execute method returns either a
    successful result or an error.
    """

    def __init__(self):
        self.handlers = []
        self.transformers = []

    def then(self, handler):
        self.handlers.append(handler)
        return self

    def intersperse(self, wrapper):
        self.transformers.append(wrapper)
        return self

    def execute(self, initial):
        """ For each handler, run them in order passing the context object to each handler.
        The handlers can update the context by returning a dict of key/value pairs for which
        to update the context object with.
        The required parameters for each handler are determined from the function parameters,
        and the appropriate parameters are passed from the context object to the handler.
        """
        def identity(f, *args, **kwargs):
            return f(*args, **kwargs)

        transformer = identity
        for t in self.transformers:
            transformer = t(transformer)

        context = dict(initial)
        # For each handler:
        #   - Acquire parameters from context
        #   - Execute handler with parameters
        #   - Update context with handler results
        for stage, handler in enumerate(self.handlers):
            handler_params = handler.__code__.co_varnames[:handler.__code__.co_argcount]
            # context.get(param) defaults to None if no value is present in context.
            acquired_params = dict()
            for param in handler_params:
                value = context.get(param)
                if value is None:
                    # Currently passing None if param is not found in context.
                    # Could throw, stop, or add a way to implement custom behavior here.
                    print(f"Could not find {param} in pipeline context!")
                acquired_params[param] = value

            print(f"Executing stage [{stage}]:  {handler.__name__}({', '.join([str(param) for param in handler_params])})")
            try:
                # Run the next handler with the appropriate parameters
                # and apply any transforms.
                result = transformer(handler, **acquired_params)
                # Update the context object with the key/value pairs from result.
                if isinstance(result, dict):
                    context.update(result)

            except Exception as e:
                print(f"Pipeline handler [{stage}] {handler.__name__} failed with exception.")
                raise e

        return context


def handle_read_las_data(input_file_path):
    # Construct and execute reader pipeline.
    json_pipeline = [
        {
            "type": "readers.las",
            "filename": input_file_path
        }
    ]
    pdal_pipeline = pdal.Pipeline(json.dumps(json_pipeline))
    pdal_pipeline.execute()

    # Gather required data (metadata + point data) from processed .las file.
    pdal_metadata = pdal_pipeline.metadata["metadata"]["readers.las"]
    header_keys = [
        "scale_x", "scale_y", "scale_z",
        "offset_x", "offset_y", "offset_z",
        "minx", "miny", "minz",
        "maxx", "maxy", "maxz",
        "count"
    ]
    header = {key: pdal_metadata[key] for key in header_keys}
    pdal_data = pdal_pipeline.arrays[0]

    # Structure the data from PDAL to be used by the rest of our program.
    point_count = header["count"]
    scale_xyz = np.array([header["scale_x"], header["scale_y"], header["scale_z"]])
    min_xyz = (np.array([header["minx"], header["miny"], header["minz"]]) / scale_xyz).astype("int")
    max_xyz = (np.array([header["maxx"], header["maxy"], header["maxz"]]) / scale_xyz).astype("int")
    bounds_xyz = (min_xyz, max_xyz)
    range_xyz = (max_xyz - min_xyz) * scale_xyz

    # Coordinates are represented using scaled floats.
    # Want to keep arithmetic within integers for the algorithm.
    xs = (pdal_data["X"] / scale_xyz[0]).astype("int")
    ys = (pdal_data["Y"] / scale_xyz[2]).astype("int")
    zs = (pdal_data["Z"] / scale_xyz[2]).astype("int")

    points_xyz = np.array([xs, ys, zs])

    return {
        "points_xyz": points_xyz,
        "point_count": point_count,
        "bounds_xyz": bounds_xyz,
        "range_xyz": range_xyz,
        "scale_xyz": scale_xyz,
        "min_xyz": min_xyz,
        "max_xyz": max_xyz
    }


def handle_las2img(points_xyz, bounds_xyz, range_xyz, scale_xyz, discretization, resolution):
    grid_size = np.ceil(range_xyz[:2] / resolution).astype("int")
    cell_size = np.ceil(resolution / scale_xyz[:2]).astype("int")
    grid = las2img(points_xyz, bounds_xyz, grid_size, cell_size, discretization)

    return {
        "grid": grid,
        "grid_size": grid_size,
        "cell_size": cell_size
    }


def handle_compute_patches(grid, discretization, min_height, neighbor_mask):
    all_patches = compute_patches(grid, discretization, min_height, neighbor_mask)
    # labeled_grid = gaussian_filter(labeled_grid, sigma=0.4)

    return {
        "all_patches": all_patches
    }


def handle_patches_to_dict(all_patches):
    patches_dict = patches_to_dict(all_patches)
    return {
        "patches_dict": patches_dict
    }


def handle_compute_patches_labeled_grid(grid, all_patches):
    labeled_grid = create_labeled_grid(grid, all_patches)
    return {
        "labeled_grid": labeled_grid
    }


def handle_compute_patch_neighbors(grid, labeled_grid, all_patches):
    compute_patch_neighbors(grid, labeled_grid, all_patches)


def handle_compute_hierarchies(all_patches, patches_dict):
    hierarchies= compute_hierarchies(all_patches, patches_dict)

    return {
        "hierarchies": hierarchies,
    }


def handle_find_connected_hierarchies(patches_dict):
    connected_hierarchies = find_connected_hierarchies(patches_dict)

    return {
        "connected_hierarchies": connected_hierarchies
    }


def handle_calculate_edge_weight(hierarchies, connected_hierarchies, weights):
    hdag = calculate_edge_weight(hierarchies, connected_hierarchies, weights)


    return {
        "hdag": hdag
    }


def handle_partition_graph(hdag, weight_threshold, patches_dict):
    trees = partition_graph(hdag, weight_threshold, patches_dict)

    return {
        "trees": trees
    }


def handle_trees_to_labeled_grid(trees, grid_size):
    x_size, y_size = grid_size
    labeled_partitions = trees_to_labeled_grid(trees, x_size, y_size)

    return {
        "labeled_partitions": labeled_partitions
    }


def handle_partitions_to_labeled_grid(partitioned_graph, grid_size):
    labeled_partitions = partitions_to_labeled_grid(partitioned_graph, grid_size[0], grid_size[1])
    # labeled_partitions = gaussian_filter(labeled_partitions, sigma=1)

    return {
        "labeled_partitions": labeled_partitions
    }


def handle_adjust_partitions(patches_dict, labeled_partitions, hp_map):
    labeled_partitions = adjust_partitions(patches_dict, labeled_partitions, hp_map)

    return {
        "labeled_partitions": labeled_partitions
    }


def handle_label_points(labeled_partitions, points_xyz, point_count, min_xyz, cell_size):
    labeled_points = laslabel(labeled_partitions, points_xyz, point_count, min_xyz, cell_size)

    return {
        "labeled_points": labeled_points
    }


def handle_save_tree_raster(save_tree_raster, tree_raster_save_path, partitioned_graph, labeled_partitions, grid, input_file_path, resolution, discretization):
    if not save_tree_raster:
        return

    for index, partition in enumerate(partitioned_graph):
        id = partition.id
        if id == 0:
            continue

        col, row = np.indices((labeled_partitions.shape[1], labeled_partitions.shape[0]))
        mask = labeled_partitions[row, col] == id

        channel = ((1 - grid / discretization) * 255).transpose().astype("uint8")
        r = (channel % 256).astype("uint8")
        g = (channel % 256).astype("uint8")
        b = (channel % 256).astype("uint8")

        r[mask] = 255

        image_color = np.dstack((r, g, b))
        img = Image.fromarray(image_color, "RGB")

        os.makedirs(tree_raster_save_path, exist_ok=True)
        file_name = os.path.split(input_file_path)[1]
        save_name = f"T{id}_{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
        full_path = os.path.join(tree_raster_save_path, save_name)
        img.save(full_path)

    print(f"    - Saved tree rasters to \"{tree_raster_save_path}\"")


def handle_save_partition_raster(save_partition_raster, partition_raster_save_path, discretization, grid, labeled_partitions, input_file_path, resolution):
    if not save_partition_raster:
        return

    labels = labeled_partitions.transpose()
    r = ((labels * 7) % 256).astype("uint8")
    g = ((labels * 13) % 256).astype("uint8")
    b = ((labels * 23) % 256).astype("uint8")
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    os.makedirs(partition_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(partition_raster_save_path, save_name)
    img.save(full_path)

    print(f"    - Saved partition raster to \"{full_path}\"")

    return {
        "png_raster_path": full_path
    }


def handle_save_patches_raster(save_patches_raster, patches_raster_save_path, labeled_grid, input_file_path, resolution, discretization):
    if not save_patches_raster:
        return

    labels = labeled_grid.transpose()
    r = ((labels * 7) % 256).astype("uint8")
    g = ((labels * 13) % 256).astype("uint8")
    b = ((labels * 23) % 256).astype("uint8")
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    os.makedirs(patches_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(patches_raster_save_path, save_name)
    img.save(full_path)

    print(f"    - Saved patches raster to \"{full_path}\"")


def handle_save_grid_raster(save_grid_raster, grid_raster_save_path, grid, input_file_path, resolution, discretization):
    if not save_grid_raster:
        return

    image_gray = ((1 - grid / discretization) * 255).transpose().astype("uint8")
    img = Image.fromarray(image_gray)

    os.makedirs(grid_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(grid_raster_save_path, save_name)
    img.save(full_path)

    print(f"    - Saved grid raster to \"{full_path}\"")


def handle_save_centroids_raster(save_centroids_raster, centroids_raster_save_path, hierarchies, grid, input_file_path, resolution, discretization):
    if not save_centroids_raster:
        return

    image_gray = ((1 - grid / discretization) * 255).transpose().astype("uint8")
    r = image_gray
    g = image_gray
    b = image_gray

    root_patches = [h.root.patch for h in hierarchies]
    centroids = [p.centroid for p in root_patches]
    for y, x in centroids:
        x, y = int(x), int(y)
        r[x, y] = 255
        g[x, y] = 255
        b[x, y] = 0

    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    os.makedirs(centroids_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(centroids_raster_save_path, save_name)
    img.save(full_path)

    print(f"    - Saved centroids raster to \"{full_path}\"")


def handle_label_point_cloud(input_file_path, bounds_xyz, scale_xyz, partition_raster_save_path, point_cloud_save_path, png_raster_path):
    min_xyz, max_xyz = bounds_xyz
    scale_x, scale_y, scale_z = scale_xyz

    # min/max x/y for passing to gdal_translate
    min_x = min_xyz[0] * scale_x
    max_x = max_xyz[0] * scale_x
    min_y = min_xyz[1] * scale_y
    max_y = max_xyz[1] * scale_y

    os.makedirs(point_cloud_save_path, exist_ok=True)

    t = time.localtime()
    current_time = time.strftime("%Y_%m_%d_%H%M%S", t)

    # Output with date/time string appended
    gtiff_output_path = os.path.join(partition_raster_save_path, f"output_{current_time}.tif")
    las_output_path = os.path.join(point_cloud_save_path, f"output_{current_time}.las")

    # spawn gdal_translate program to translate png to GeoTiff
    translate_command = f"gdal_translate -of GTiff -a_srs EPSG:4326 -a_ullr {min_x} {min_y} {max_x} {max_y} {png_raster_path} {gtiff_output_path}"
    os.system(translate_command)

    print(f"saved gtiff to {gtiff_output_path}")

    # Overlay the raster onto the point cloud using the pdal pipeline
    pipeline = pdal.Reader.las(filename=input_file_path) | pdal.Filter.colorization(raster=gtiff_output_path, dimensions="Red, Green, Blue")
    pipeline.execute()

    # Write the colored/filtered points to the file
    colored_points = pipeline.arrays[0]
    pipeline = pdal.Writer.las(filename=las_output_path, dataformat_id=2, a_srs="EPSG:4326").pipeline(colored_points)
    pipeline.execute()


def print_runtime(f):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f"    - Finished in {elapsed:.2f} seconds")
        return result
    return wrapper


def run_algo(user_data):
    algorithm = Pipeline() \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_save_grid_raster) \
        .then(handle_compute_patches) \
        .then(handle_patches_to_dict) \
        .then(handle_compute_patches_labeled_grid) \
        .then(handle_compute_patch_neighbors) \
        .then(handle_save_patches_raster) \
        .then(handle_compute_hierarchies) \
        .then(handle_save_centroids_raster) \
        .then(handle_find_connected_hierarchies) \
        .then(handle_calculate_edge_weight) \
        .then(handle_partition_graph) \
        .then(handle_trees_to_labeled_grid)

    # handle_save_labeled_grid_as_image has an if checking for should_save as well,
    # so having both ifs is redundant. Doing this to show that there is a lot of
    # flexibility in how the Pipeline and its components are used.
    if user_data["save_partition_raster"]:
        algorithm.then(handle_save_partition_raster) \
        .then(handle_label_points) \
        .then(handle_save_tree_raster)

    algorithm.then(handle_label_point_cloud)
    algorithm.intersperse(print_runtime)

    result = algorithm.execute(user_data)
    print("== Result")
    print(result.keys())
    print("== Tree Count")
    print(len(result["partitioned_graph"]))

    return result
