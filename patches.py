# patches.py
#
# Description:
#   Transforms a LAS file into an aerial-view grayscale image.
#   The lowest points are black, the highest points are white.
#   Points for which there is no collected data are colored black.

from las2img import las2img


if __name__ == "__main__":
    file_path = "./data/easy_nno.las"
    resolution = 0.5
    discretization = 32
    grid = las2img(file_path, resolution, discretization)
    print(grid)
