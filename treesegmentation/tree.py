#tree.py
import numpy as np


class Tree:
    def __init__(self, root, hierarchies):
        self.root = root
        self.id = root.root_id
        self.cells = []
        self.hierarchies = hierarchies
        self.patches = {}

    def get_hierarchies(self):
        return self.partition.children

    def add_patch(self, patch):
        self.patches[patch.id] = patch


def partition_graph(HDAG, weight_threshold, patches_dict):
    trees = {}

    # map of hierarchies to partitions
    hp_map = {}

    hierarchy_tree_map = {}

    HDAG.remove_non_maximal_inbound_edges()
    HDAG.partition_by_weight_threshold(weight_threshold)
    hdag_nodes = HDAG.nodes.copy()
    # get parentless source nodes, these are the trees
    sources = HDAG.get_source_nodes()

    # TODO: See about making this a method for Partition class
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
    labeled_tree_grid = np.zeros(shape=(x_size, y_size), dtype=np.int64)

    for tree_id, tree in trees.items():
        #print(np.array(tree.cells))
        idx = np.array(tree.cells)
        #print()
        #labeled_tree_grid[[np.array(tree.cells)]] = tree_id
        labeled_tree_grid[idx[:, 0], idx[:, 1]] = tree_id
    #print(labeled_tree_grid)
    return labeled_tree_grid

# Old way
'''
def partitions_to_trees(partitions, labeled_partitions):
    trees = []
    for partition in partitions:
        tree_id = partition.id
        cells = np.argwhere(labeled_partitions == tree_id)
        tree = Tree(partition, cells)
        trees.append(tree)

    return trees
'''