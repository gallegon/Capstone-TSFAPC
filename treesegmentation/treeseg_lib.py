import laspy
import numpy as np
import os.path
from PIL import Image

from .hdag import *
from .hierarchy import *
from .las2img import *
from .patch import *


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

    def then(self, handler):
        self.handlers.append(handler)
        return self

    def execute(self, initial):
        """ For each handler, run them in order passing the context object to each handler.
        The handlers can update the context by returning a dict of key/value pairs for which
        to update the context object with.
        The required parameters for each handler are determined from the function parameters,
        and the appropriate parameters are passed from the context object to the handler.
        """
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

            # print(f"Executing stage [{stage}]:  {handler.__name__}({', '.join([str(k) + '=' + str(v) for k, v in acquired_params.items()])})")
            print(f"Executing stage [{stage}]:  {handler.__name__}({', '.join([str(param) for param in handler_params])})")
            try:
                # Run the next handler with the appropriate parameters.
                result = handler(**acquired_params)
                # Update the context object with the key/value pairs from result.
                if isinstance(result, dict):
                    context.update(result)

            except Exception as e:
                print(f"Pipeline handler [{stage}] {handler.__name__} failed with exception.")
                raise e

        return context


def handle_read_las_data(input_file_path):
    data = laspy.read(input_file_path)
    header = data.header
    point_count = header.point_count
    scale_xyz = header.scales
    # offset_xyz = header.offsets # Offsets currently not being used...
    # Keeping the mins and maxs unscaled, making them ints and not floats.
    # Doing this because laspy sample_data.X, .Y, and .Z are all unscaled ints.
    # However *_min and *_max are all scaled.
    min_xyz = (np.array([header.x_min, header.y_min, header.z_min]) / scale_xyz).astype("int")
    max_xyz = (np.array([header.x_max, header.y_max, header.z_max]) / scale_xyz).astype("int")
    bounds_xyz = (min_xyz, max_xyz)
    range_xyz = (max_xyz - min_xyz) * scale_xyz
    points_xyz = np.array([data.X, data.Y, data.Z])
    
    return {
        "points_xyz": points_xyz,
        "point_count": point_count,
        "bounds_xyz": bounds_xyz,
        "range_xyz": range_xyz,
        "scale_xyz": scale_xyz
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
    labeled_grid = create_labeled_grid(grid, all_patches)
    # labeled_grid = gaussian_filter(labeled_grid, sigma=0.4)
    compute_patch_neighbors(grid, labeled_grid, all_patches)

    return {
        "all_patches": all_patches,
        "labeled_grid": labeled_grid
    }


def handle_compute_hierarchies(all_patches):
    hierarchies, contact = compute_hierarchies(all_patches)

    return {
        "hierarchies": hierarchies,
        "contact": contact
    }


def handle_find_connected_hierarchies(contact, hierarchies, weights):
    connected_hierarchies = find_connected_hierarchies(contact)
    hdag = calculate_edge_weight(hierarchies, connected_hierarchies, weights)

    return {
        "connected_hierarchies": connected_hierarchies,
        "hdag": hdag
    }


def handle_partition_graph(hdag, weight_threshold):
    partitioned_graph = partition_graph(hdag, weight_threshold)

    return {
        "partitioned_graph": partitioned_graph
    }


def handle_partitions_to_labeled_grid(partitioned_graph, grid_size):
    labeled_partitions = partitions_to_labeled_grid(partitioned_graph, grid_size[0], grid_size[1])
    # labeled_partitions = gaussian_filter(labeled_partitions, sigma=1)

    return {
        "labeled_partitions": labeled_partitions
    }


def handle_save_labeled_raster(save_labeled_raster, labeled_raster_save_path, discretization, grid, labeled_partitions, input_file_path, resolution):
    if not save_labeled_raster:
        return
    
    print(f"Saving labeled raster to \"{labeled_raster_save_path}\"")
    image_grayscale = ((1 - grid / discretization) * 255).transpose().astype("uint8")
    r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()

    for x, y in np.ndindex(len(grid[0]), len(grid[1])):
        # if labeled_partitions[x][y] != 0:
        r[x, y] = labeled_partitions[x][y] % 256
        g[x, y] = (labeled_partitions[x][y] * 2) % 256
        b[x, y] = (labeled_partitions[x][y] * 4) % 256

    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")
    os.makedirs(labeled_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(labeled_raster_save_path, save_name)
    print(f"Saved labeled raster to \"{full_path}\"")
    img.save(full_path)


def handle_save_patches_raster(save_patches_raster, patches_raster_save_path, labeled_grid, all_patches, input_file_path, resolution, discretization):
    if not save_patches_raster:
        return
    
    print(f"Saving patches raster to \"{patches_raster_save_path}\"")
    # Format the sample_data into an image appropriate format for PIL.
    # Dear god this part needs to be written literally any other way.
    import random
    labeled_raster = labeled_grid.transpose()
    image_grayscale = labeled_raster.astype("uint8")
    r, g, b = image_grayscale.copy(), image_grayscale.copy(), image_grayscale.copy()
    rmap = {p.id: random.randint(0, 200) for p in all_patches}
    gmap = {p.id: random.randint(0, 200) for p in all_patches}
    bmap = {p.id: random.randint(0, 200) for p in all_patches}
    for i in range(labeled_grid.shape[1]):
        for j in range(labeled_grid.shape[0]):
            id = labeled_raster[i, j]
            if id not in rmap or id == 0:
                r[i, j] = 255
                g[i, j] = 255
                b[i, j] = 255
            else:
                r[i, j] = rmap[id]
                g[i, j] = gmap[id]
                b[i, j] = bmap[id]
    
    image_color = np.dstack((r, g, b))
    img = Image.fromarray(image_color, "RGB")

    os.makedirs(patches_raster_save_path, exist_ok=True)
    file_name = os.path.split(input_file_path)[1]
    save_name = f"{file_name}_{resolution}-{discretization}-{img.width}x{img.height}.png"
    full_path = os.path.join(patches_raster_save_path, save_name)
    img.save(full_path)
    print(f"Saved patches raster to \"{full_path}\"")


def run_algo(user_data):
    algorithm = Pipeline() \
        .then(handle_read_las_data) \
        .then(handle_las2img) \
        .then(handle_compute_patches) \
        .then(handle_save_patches_raster) \
        .then(handle_compute_hierarchies) \
        .then(handle_find_connected_hierarchies) \
        .then(handle_partition_graph) \
        .then(handle_partitions_to_labeled_grid) \
        .then(handle_save_labeled_raster)

    result = algorithm.execute(user_data)
    return result
