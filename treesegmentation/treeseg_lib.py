import json
import os.path
import time
import timeit
from pathlib import Path

import pdal
from scipy.ndimage import gaussian_filter
from PIL import Image

from .hdag import *
from .hierarchy import *
from .las2img import *
from .patch import *
from .tree import *


class Pipeline:
    """ Controls the order and handling of multiple stages of a process.

    Build a pipeline by constructing a Pipeline and using the ``.then`` method
    to add stages.

    Note:
        Use the ``.execute`` method to pass an initial input and begin running the
        first stage. Each stage is executed sequentially, passing output from
        the previous stage as input to the next stage.

        Either every stage completes successfully, or somewhere along the line
        an error occurs. The ``.execute`` method returns either a
        successful result or an error.
    """

    def __init__(self, verbose=False):
        self.handlers = []
        self.transformers = []
        self.verbose = True if verbose else False

    def then(self, handler):
        """Adds the next sequential stage in this pipeline.

        :param handler: Handler function to be executed.
            See ``Pipeline.execute`` for specification of handler functions.

        :return: Returns this pipeline. Allows ``.then`` calls to be chained.
        """
        self.handlers.append(handler)
        return self

    def intersperse(self, wrapper):
        """Call the ``wrapper`` function on each stage of the pipeline.

        :param wrapper: Decorator like function to be applied to each handler function before execution.

        :return: Returns this pipeline. Allows ``.intersperse`` calls to be chained.
        """
        self.transformers.append(wrapper)
        return self

    def execute(self, initial):
        """ Run each handler in order passing the context object to each handler.

        The required parameters for each handler are determined from the function definition,
        and the appropriate parameters are passed from the context object to the handler upon execution.

        Handler functions can update the context by returning a dict of key/value pairs for which
        to update the context object with. Although this is not required.

        :param initial: The initial context dictionary to be updated after each stage in the pipeline.

        :return: The resulting context dictionary (string names to values).
        """

        def identity(f, *args, **kwargs):
            return f(*args, **kwargs)

        # Construct the transformer to apply to each handler function.
        # Start with the identity transformer so the last application can be on
        # the handler function itself, beginning the execution.
        transformer = identity
        for t in self.transformers:
            transformer = t(transformer)

        context = dict(initial)
        # For each handler:
        #   - Acquire parameters from context
        #   - Execute handler with parameters
        #   - Update context with handler results
        start_time = time.time()
        for stage, handler in enumerate(self.handlers):
            handler_params = handler.__code__.co_varnames[:handler.__code__.co_argcount]
            default_values = tuple() if handler.__defaults__ is None else handler.__defaults__
            default_params = dict(zip(handler_params[-len(default_values):], default_values))

            acquired_params = dict()
            for param in handler_params:
                if param in context:
                    acquired_params[param] = context[param]
                elif param in default_params:
                    acquired_params[param] = default_params[param]
                elif param == "_context":
                    acquired_params[param] = context
                else:
                    # Requested parameter is not in context, not defaulted, and not a special parameter.
                    # Currently, passing None if param is neither defaulted not found in context.
                    # Could throw, stop, or add a way to implement custom behavior here.
                    print(f"Could not find '{param}' in pipeline context!")
                    acquired_params[param] = None

            if self.verbose:
                param_list = map(lambda p: str(p) if p not in default_params else f"{p}={default_params[p]}", handler_params)
                print(f"Executing stage [{stage}]: {handler.__name__}({', '.join(param_list)})")
            else:
                print(f"Executing stage [{stage}]: {handler.__name__}")

            try:
                # Run the next handler with the appropriate parameters and apply any transforms.
                result = transformer(handler, **acquired_params)
                # Update the context object with the key/value pairs from result.
                if isinstance(result, dict):
                    context.update(result)

            except Exception as e:
                print(f"Pipeline handler [{stage}] {handler.__name__} failed with exception.")
                raise e

            if self.verbose:
                print()

        elapsed_time = time.time() - start_time

        print(f"Pipeline completed {len(self.handlers)} stages in {elapsed_time} seconds.")
        return context


def transform_print_runtime(f):
    """Wrapper/decorator which prints the runtime of the funtion when called.

    :param f: Function to print the execution time of.

    :return: Decorator applied to the given function.
    """

    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f"    - Finished in {elapsed:.2f} seconds")
        return result
    return wrapper


# ============================================================================
# = All of the following function definitions are handlers for the Pipeline. =
# ============================================================================


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


def handle_create_file_names_and_paths(input_file_path, output_folder_name=None, save_folder_name=None):
    output_folder_name = "treesegmentation_output" if output_folder_name is None else output_folder_name
    input_file_name = ".".join(os.path.split(input_file_path)[1].split(".")[:-1])
    save_folder_name = input_file_name + "_" + hex(hash(time.time()))[2:] if save_folder_name is None else save_folder_name
    output_folder_path = os.path.join(os.getcwd(), output_folder_name, save_folder_name)

    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    return {
        "output_folder_path": output_folder_path,
        "input_file_name": input_file_name
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


def handle_gaussian_filter(grid, gaussian_sigma=None):
    if gaussian_sigma is None:
        return

    grid = gaussian_filter(grid, sigma=gaussian_sigma)

    return {
        "grid": grid
    }


def handle_grid_height_cutoff(grid, min_height):
    grid[grid < min_height] = 0

    return {
        "grid": grid
    }


def handle_compute_patches(grid, discretization, min_height, neighbor_mask=None):
    all_patches = compute_patches(grid, discretization, min_height, neighbor_mask)

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


def handle_calculate_edge_weight(hierarchies, connected_hierarchies, weight_level_depth,
                                 weight_node_depth, weight_shared_ratio,
                                 weight_top_distance, weight_centroid_distance):
    weights = np.array([
        weight_level_depth,
        weight_node_depth,
        weight_shared_ratio,
        weight_top_distance,
        weight_centroid_distance
    ], dtype=np.float32)
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


def handle_save_partition_raster(input_file_name, output_folder_path, labeled_partitions, save_partition_raster=True):
    if not save_partition_raster:
        return

    labels = labeled_partitions.transpose()
    r = ((labels * 7) % 256).astype("uint8")
    g = ((labels * 13) % 256).astype("uint8")
    b = ((labels * 23) % 256).astype("uint8")
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    save_name = f"{input_file_name}_partitions.png"
    save_path = os.path.join(output_folder_path, save_name)
    img.save(save_path)

    print(f"    - Saved partition raster to \"{save_path}\"")

    return {
        "png_raster_path": save_path
    }


def handle_save_patches_raster(input_file_name, output_folder_path, labeled_grid, save_patches_raster=False):
    if not save_patches_raster:
        return

    labels = labeled_grid.transpose()
    r = ((labels * 7) % 256).astype("uint8")
    g = ((labels * 13) % 256).astype("uint8")
    b = ((labels * 23) % 256).astype("uint8")
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    save_name = f"{input_file_name}_patches.png"
    save_path = os.path.join(output_folder_path, save_name)
    img.save(save_path)

    print(f"    - Saved patches raster to \"{save_path}\"")


def handle_save_grid_raster(input_file_name, output_folder_path, grid, discretization, save_grid_raster=False):
    if not save_grid_raster:
        return

    image_gray = ((1 - grid / discretization) * 255).transpose().astype("uint8")
    img = Image.fromarray(image_gray)

    save_name = f"{input_file_name}_grid.png"
    save_path = os.path.join(output_folder_path, save_name)
    img.save(save_path)

    print(f"    - Saved grid raster to \"{save_path}\"")


def handle_label_point_cloud(input_file_path, input_file_name, output_folder_path, png_raster_path, bounds_xyz, scale_xyz, espg_string):
    if not png_raster_path:
        return

    min_xyz, max_xyz = bounds_xyz
    scale_x, scale_y, scale_z = scale_xyz

    # min/max x/y for passing to gdal_translate
    min_x = min_xyz[0] * scale_x
    max_x = max_xyz[0] * scale_x
    min_y = min_xyz[1] * scale_y
    max_y = max_xyz[1] * scale_y

    # t = time.localtime()
    # current_time = time.strftime("%Y_%m_%d_%H%M%S", t)

    # Output with date/time string appended
    gtiff_output_name = f"{input_file_name}_geotiff.tif"
    las_output_name = f"{input_file_name}_labeled.las"

    gtiff_output_path = os.path.join(output_folder_path, gtiff_output_name)
    las_output_path = os.path.join(output_folder_path, las_output_name)

    # spawn gdal_translate program to translate png to GeoTiff
    translate_command = f'gdal_translate -of GTiff -a_srs {espg_string} -a_ullr {min_x} {min_y} {max_x} {max_y} "{png_raster_path}" "{gtiff_output_path}"'
    print("+++++++++++++++")
    print(translate_command)
    print("+++++++++++++++")
    os.system(translate_command)

    # run gdal_edit to reproject into a different CRS
    edit_command = f'gdal_edit -a_srs {espg_string} "{gtiff_output_path}"'
    os.system(edit_command)

    print(f"    -- Saved GeoTIFF to {gtiff_output_path}")

    # Overlay the raster onto the point cloud using the pdal pipeline
    pipeline = pdal.Reader.las(filename=input_file_path) | pdal.Filter.colorization(raster=gtiff_output_path, dimensions="Red, Green, Blue")
    pipeline.execute()

    # Write the colored/filtered points to the file
    colored_points = pipeline.arrays[0]
    pipeline = pdal.Writer.las(filename=las_output_path, dataformat_id=2, a_srs="EPSG:4326").pipeline(colored_points)
    pipeline.execute()


def handle_save_context_file(_context, input_file_name, output_folder_path, save_context_file=False):
    if not save_context_file:
        return

    writeable_values = {}
    for key, value in _context.items():
        try:
            json.dumps(value)
            writeable_values[key] = value
        except TypeError:
            pass

    file_path = os.path.join(output_folder_path, f"{input_file_name}_pipeline.json")
    with open(file_path, "w") as file:
        json.dump(writeable_values, file, indent=4, skipkeys=True)
    print(f"    -- Saved context file to {file_path}")
