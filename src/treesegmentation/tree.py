#tree.py
import numpy as np


class Tree:
    def __init__(self, partition, cells):
        self.cells = cells
        self.partition = partition
        self.id = partition.id

    def get_hierarchies(self):
        return self.partition.children


def partitions_to_trees(partitions, labeled_partitions):
    trees = []
    for partition in partitions:
        tree_id = partition.id
        cells = np.argwhere(labeled_partitions == tree_id)
        tree = Tree(partition, cells)
        trees.append(tree)

    return trees
