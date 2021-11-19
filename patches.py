# patches.py
#
# Description:
#   Transforms a LAS file into an aerial-view grayscale image.
#   The lowest points are black, the highest points are white.
#   Points for which there is no collected data are colored black.

from scipy.ndimage import find_objects, generate_binary_structure

from las2img import las2img

if __name__ == "__main__":
    file_path = "./data/easy_nno.las"
    resolution = 10
    discretization = 32
    grid = las2img(file_path, resolution, discretization)
    print("== Discretized Grid")
    print(grid)
    height_query = 24
    print(f"== Query Height")
    print(height_query)
    containing_rect = find_objects(grid)[height_query - 1]
    print("== Containing Rect")
    print(containing_rect)
    print(f"== Object label {height_query}")
    print(grid[containing_rect] == height_query)
    # print(f"Found {num_labels} labels!")
    # print(labels)
