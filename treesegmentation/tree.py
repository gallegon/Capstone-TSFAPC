#tree.py
import numpy as np


class Tree:
    def __init__(self, root, hierarchies):
        self.root = root
        self.id = root.root_id
        self.cells = []
        self.hierarchies = hierarchies
        self.patches = {}

    def add_patch(self, patch):
        """
        Add a patch from the grid to a tree.

        :param patch: patch object
        :return: None
        """
        self.patches[patch.id] = patch


def partition_graph(HDAG, weight_threshold, patches_dict):
    """
    Partition the graph based on a weight threshold.  Assign cells and patches to trees
    based on the HDAG partitioning.

    :param HDAG: Hdag object
    :param weight_threshold: float
    :param patches_dict: dictionary of patches
    :return: array of tree objects
    """
    trees = {}

    # map of hierarchies to partitions
    hp_map = {}

    hierarchy_tree_map = {}

    # Partition the HDAG
    HDAG.remove_non_maximal_inbound_edges()
    HDAG.partition_by_weight_threshold(weight_threshold)
    hdag_nodes = HDAG.nodes.copy()

    # get parentless source nodes, these are the trees
    sources = HDAG.get_source_nodes()

    # get the reachable hierarchies from each source and add it to the source
    for source in sources:
        hierarchies = {}
        queued_nodes = {source}
        while len(queued_nodes) != 0:
            next_queue = set()
            for node in queued_nodes:
                hierarchies[node] = hdag_nodes[node]
                for child in hdag_nodes[node].children:
                    next_queue.add(hdag_nodes[node].children[child].node.id)
            queued_nodes = set(next_queue)

        # append the newly created partition to the list of source nodes
        tree = Tree(hdag_nodes[source].root, hierarchies)
        trees[tree.id] = tree

        # create a map of hierarchies and partitions
        for hierarchy in hierarchies:
            hp_map[hierarchy] = tree.id
            hierarchy_tree_map[hierarchy] = tree.id

    # Return an array of the partitions
    for patch_id, patch in patches_dict.items():
        hierarchy_id = patch.nearest_hierarchy_id
        associated_tree = trees[hp_map[hierarchy_id]]
        cells_to_add = patch.cells.tolist()
        associated_tree.cells += cells_to_add
        associated_tree.add_patch(patch)
    return trees


def trees_to_labeled_grid(trees, x_size, y_size):
    """
    From an array of tree objects, label a 2D grid with their unique IDs.

    :param trees: list of tree objects
    :param x_size: int
    :param y_size: int
    :return: 2D grid of labeled trees.
    """
    # Create a grid to label
    labeled_tree_grid = np.zeros(shape=(x_size, y_size), dtype=np.int64)

    # For every tree's cells, label that cell with the tree id in labeled grid
    for tree_id, tree in trees.items():
        idx = np.array(tree.cells)
        labeled_tree_grid[idx[:, 0], idx[:, 1]] = tree_id
    return labeled_tree_grid
