import laspy
import sys

print(sys.argv)
las = laspy.read(sys.argv[1])
"""
print(las.header.scale)
print("Xmin: " + str(las.header.x_min / 0.01) + " Xmax: " + str(las.header.x_max))
print("Ymin: " + str(las.header.y_min / 0.01) + " Ymax: " + str(las.header.y_max))
"""

# get the relative values of start and end (adjusted by scale)
x_start = int(las.header.x_min / las.header.scale[0])
y_start = int(las.header.y_min / las.header.scale[1])

# get the max values (adjusted by the scale)
x_end = int(las.header.x_max / las.header.scale[0])
y_end = int(las.header.y_max / las.header.scale[1])

# cell size in meters, needs to be divided by the scale defined by file header
cell_size = int(sys.argv[2])
cell_size = int(cell_size / las.header.scale[0])

grid_size_x = (x_end - x_start) // cell_size
if ((x_end - x_start) % cell_size != 0):
    grid_size_x += 1

grid_size_y = (y_end - y_start) // cell_size
if ((y_end - y_start) % cell_size != 0):
    grid_size_y += 1

# initialize array with point records with X, Y, Z height of -1 (invalid)
p = las.point_format
p.X = -1
p.Y = -1
p.Z = -1
arr = [[p]*grid_size_x for i in range(grid_size_y)]

for points in las:
    x_idx = (points.X - x_start) // cell_size
    y_idx = (points.Y - y_start) // cell_size
    if points.Z > arr[y_idx][x_idx].Z:
        arr[y_idx][x_idx] = points

# print the array nicely
buffer = [0 for i in range(len(arr[0]))]
for i in range(0, len(arr)):
    for j in range(0, len(arr[0])):
        buffer[j] = arr[i][j].Z# * las.header.scale[2]
        if j == len(arr) - 1:
            print(buffer)

print("Highest point in cell (" + sys.argv[3] + ", " + sys.argv[4] + "):")
print(str(arr[int(sys.argv[4])][int(sys.argv[3])].Z * las.header.scale[2]) + "m")

