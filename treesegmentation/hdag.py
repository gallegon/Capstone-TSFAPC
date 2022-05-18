# hdag.py
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

import numpy as np
import itertools


class Hdag:
    def __init__(self):
        self.nodes = {}

    def initialize_from_hierarchies(self, hierarchies):
        """
        Initialize the hierarchy directed acyclic graph from a list of generated
        hierarchies.
        :param hierarchies: List of Hierarchy objects
        :return: None
        """
        for hierarchy in hierarchies:
            self.nodes[hierarchy.root_id] = HdagNode(hierarchy)

    def add_edge(self, parent_id, child_id, weight):
        parent_edge = HdagEdge(self.nodes[parent_id], weight)
        child_edge = HdagEdge(self.nodes[child_id], weight)
        self.nodes[parent_id].add_child(child_edge)
        self.nodes[child_id].add_parent(parent_edge)

    def remove_edge(self, parent_id, child_id):
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        parent.remove_child(child_id)
        child.remove_parent(parent_id)

    def get_hdag_node(self, node_id):
        return self.nodes[node_id]

    def remove_non_maximal_inbound_edges(self):
        # Partition based on non-maximal inbound edges
        for node_id, node in self.nodes.items():
            #
            max_parent_weight = -1
            max_parent_key = -1

            # Choose the edges to remove
            non_maximal_inbound_edges = []
            for parent_id, parent in node.parents.items():
                weight = parent.weight
                if weight > max_parent_weight:
                    non_maximal_inbound_edges.append(max_parent_key)
                    max_parent_weight = weight
                    max_parent_key = parent_id
                else:
                    non_maximal_inbound_edges.append(parent_id)

            # remove every edge except the maximum parent
            for edge in non_maximal_inbound_edges:
                if edge != -1:
                    self.remove_edge(edge, node_id)

    def partition_by_weight_threshold(self, weight_threshold):
        for node_id, node in self.nodes.items():
            edge_to_remove = -1
            for parent_id, parent in node.parents.items():
                #parent_edge = hdag_node.parents[parent]
                weight = parent.weight
                if weight < weight_threshold:
                    edge_to_remove = parent_id

            if edge_to_remove != -1:
                self.remove_edge(parent_id, node_id)

    def get_source_nodes(self):
        source_nodes = []
        for node_id, node in self.nodes.items():
            if len(node.parents) == 0:  # No parents, source node
                source_nodes.append(node_id)
        return source_nodes


class HdagEdge:
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight


class HdagNode:
    def __init__(self, root):
        self.id = root.root_id
        self.root = root
        self.parents = {}  # HdagEdges
        self.children = {}  # HdagEdges

    # add a parent to a node
    def add_parent(self, parent):
        self.parents[parent.node.id] = parent

    def add_child(self, child):
        self.children[child.node.id] = child

    def remove_parent(self, parent_id):
        self.parents.pop(parent_id)

    def remove_child(self, child_id):
        self.children.pop(child_id)


class Partition:
    def __init__(self, root, hierarchies):
        self.id = root.root_id
        self.root = root
        self.children = hierarchies

    def get_child_cells(self, child, patch):
        return self.children[child].root.nodes_by_id[patch].patch.cells

    def get_child_nodes_by_id(self, child):
        return self.children[child].root.nodes_by_id

    def get_root_patch_cells(self):
        return self.root.root.patch.cells


def find_connected_hierarchies(patch_dict):
    contact_patches = set()
    for patch in patch_dict.values():
        hierarchies = patch.hierarchies
        if len(hierarchies) > 1:
            hierarchy_combinations = itertools.combinations(hierarchies, 2)
            for h1, h2 in hierarchy_combinations:
                h1_id = h1.root_id
                h2_id = h2.root_id
                if h1_id > h2_id:
                    contact_patches.add((h1_id, h2_id))
                else:
                    contact_patches.add((h2_id, h1_id))

    return contact_patches


# Calculate the node depth, level depth, and shared cell count for two hierarchies
def calculate_depth(h1, h2, shared_patches):
    shared_cell_count = 0

    level_depth = np.inf
    node_depth = np.inf

    for contact_patch in shared_patches:
        # Get level depth and node depth for each hierarchy
        h1_level_depth, h1_node_depth = h1.node_depths_by_id[contact_patch]
        h2_level_depth, h2_node_depth = h2.node_depths_by_id[contact_patch]

        # Calculate the minimums for each depth
        level_depth = min(level_depth, h1_level_depth, h2_level_depth)
        node_depth = min(node_depth, h1_node_depth, h2_node_depth)

        shared_cell_count += h1.nodes_by_id[contact_patch].patch.cell_count  

    return 1 / level_depth, 1 / node_depth, shared_cell_count


# Calculate edge orientation and weight for each unique heirarchy pair
def set_weight_and_orientation(h1, h2, weight, h1_cc, h2_cc, hdag):
    # IDs of the top patches for a hierarchy
    h1_id = h1.root_id
    h2_id = h2.root_id

    # See if we can assign edge direction based on height level first
    if h1.height > h2.height:
        hdag.add_edge(h1_id, h2_id, weight)
        return
    elif h2.height > h1.height:
        hdag.add_edge(h2_id, h1_id, weight)
        return
    else:
        # The following if/else trees are for determining edge direction
        # if the height level of the two hierarchies are equal.

        # Try to determine edge direction based on total hierarchy cell count
        if h1_cc > h2_cc:
            hdag.add_edge(h1_id, h2_id, weight)
            return
        elif h2_cc > h1_cc:
            hdag.add_edge(h2_id, h1_id, weight)
            return
        else:
            # Try to determine direction based on the cell count of the top patch
            # of each hierarchy
            if h1.root.patch.cell_count > h2.root.patch.cell_count:
                hdag.add_edge(h1_id, h2_id, weight)
                return
            elif h2.root.patch.cell_count > h1.root.patch.cell_count:
                hdag.add_edge(h2_id, h1_id, weight)
                return
            else:
                # Finally try to determine direction based on the location of each hierarchy's top patch
                # centroid. First look for the "bottom most" centroid (highest 'i' value), then look for
                # the "right most" centroid (highest 'j' value).
                result = h1.root.patch.centroid - h2.root.patch.centroid

                if result[0] == 0:
                    if result[1] > 0:
                        hdag.add_edge(h1_id, h2_id, weight)
                    else:
                        hdag.add_edge(h2_id, h1_id, weight)
                    return
                else:
                    if result[0] > 0:
                        hdag.add_edge(h1_id, h2_id, weight)
                    else:
                        hdag.add_edge(h2_id, h1_id, weight)
                    return


def calculate_edge_weight(hierarchies, connected_hierarchies, weights):
    """
    Compute statistics for a pair of connected hierarchies and calculate the
    edge weight from those statistics.  Give this weight to set_weight_and_orientation()
    to create a directed edge in the H-DAG.  Return
    :param hierarchies: List of Hierarchy objects
    :param connected_hierarchies: List of connected hierarchy pairs
    :param weights: The constants to apply to each statistic for the final weight computation
    :return: Hdag Object (Hierarchy directed acyclic graph)
    """
    hierarchy_dict = {}

    HDAG = Hdag()
    HDAG.initialize_from_hierarchies(hierarchies)

    for hierarchy in hierarchies:
        hierarchy_dict[hierarchy.root_id] = hierarchy

    for i, j in connected_hierarchies:
        h1 = hierarchy_dict[i]
        h2 = hierarchy_dict[j]

        shared_patches = np.intersect1d(h1.nodes_by_id_array, h2.nodes_by_id_array)
        level_depth, node_depth, shared_cell_count = calculate_depth(h1, h2, shared_patches)
        
        shared_ratio = shared_cell_count / (h1.cell_count + h2.cell_count - shared_cell_count)

        # Get the top patch centroids
        h1_top = h1.root.patch.centroid
        h2_top = h2.root.patch.centroid

        # Calculate top and centroid distance statistic.  Reshape top patch 
        # centroids as 2-d numpy arrays because cdist() expects a 2-d array
        top_distance = 1 / (distance.cdist(np.reshape(h1_top, (-1, 2)), 
            np.reshape(h2_top, (-1, 2)), 'euclidean')[0][0])

        centroid_distance = 1 / (distance.cdist(np.reshape(h1.height_adjusted_centroid, (-1, 2)),
            np.reshape(h2.height_adjusted_centroid, (-1, 2)), 'euclidean')[0][0])
        
        # Convert the statistics to a numpy float array
        scores = np.array([level_depth, node_depth, shared_ratio, top_distance, centroid_distance],
            dtype=np.float32)

        # Finally calculate the edge weight for h1, h2.  Find the direction of the weighted edge.
        edge_weight = np.sum((scores * weights))
        set_weight_and_orientation(h1, h2, edge_weight, h1.cell_count, h2.cell_count, HDAG)

    return HDAG


def partitions_to_labeled_grid(partitions, x, y):
    labeled_grid = np.zeros(shape=(x,y), dtype=np.int64)
    for partition in partitions:
        partition_cells = []
        partition_cells.append(partition.get_root_patch_cells())

        for child in partition.children:
            for patch in partition.get_child_nodes_by_id(child):
                partition_cells.append(partition.get_child_cells(child, patch))

        for cells in partition_cells:
            for x, y in cells:
                labeled_grid[x][y] = partition.id

    return labeled_grid



def adjust_partitions(patches_dict, labeled_partitions, hp_map):
    """
    Adjust the labeled grid so that there are no patches that belong to
    multiple trees.  After creating the directed acyclic graph, the
    graph is segmented at the hierarchy level, however not at the
    patch level.  Contact patches belong to multiple hierarchies.  This
    method assigns the patch to the closest hierarchy, making sure that a
    patch can only belong to one tree. This is based on the height-adjusted
    centroid of the hierarchy and the centroid of the contact patch.

    :param patch_dict: Dictionary of Patch objects
    :param labeled_partitions: NumPy array of integers
    :param contact:
    :param hp_map:
    :return: NumPy array of integers which are the Tree IDs
    """
    adjusted_partitions = labeled_partitions.copy()

    # for each contact patch, look for the closest hierarchy and assign it
    for contact_patch_id, associated_hierarchies in patches_dict.items():
        patch = patches_dict[contact_patch_id]

        # List comprehension method
        
        ps = [p for p in all_patches if p.id == contact_patch_id-1]
        # print(f"{ps=}")
        patch = ps[0]
        
        patch_cells = patch.cells
        centroid = patch.centroid

        min_distance = np.inf
        closest_hierarchy = None

        # Get the centroids for each hierarchy that claims the contact patch
        for hierarchy in associated_hierarchies:
            centroid_distance = (distance.cdist(np.reshape(centroid, (-1, 2)),
                                                np.reshape(hierarchy.height_adjusted_centroid, (-1, 2)),
                                                'euclidean')[0][0])

            # If the current hierarchy is closer, assign the patch to it
            if centroid_distance < min_distance:
                closest_hierarchy = hierarchy
                min_distance = centroid_distance

        # Finally adjust the partitions in the labeled grid
        partition_to_adjust = hp_map[closest_hierarchy.root_id]

        for i, j in patch_cells:
            adjusted_partitions[i][j] = partition_to_adjust

    return adjusted_partitions
