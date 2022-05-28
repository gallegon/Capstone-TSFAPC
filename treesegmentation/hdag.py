"""Module for dealing with the Hierarchy Directed Acyclic Graph (HDAG) data object.
"""

from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

import numpy as np
import itertools


class Hdag:
    def __init__(self):
        self.nodes = {}

    def initialize_from_hierarchies(self, hierarchies):
        """
        Initialize the HDAG from a list of hierarchies

        :param hierarchies: list of hierarchy objects
        :return: None
        """
        for hierarchy in hierarchies:
            self.nodes[hierarchy.root_id] = HdagNode(hierarchy)

    def add_edge(self, parent_id, child_id, weight):
        """
        Add a directed edge from a parent graph node to child

        :param parent_id: int
        :param child_id: int
        :param weight: float
        :return: None
        """
        parent_edge = HdagEdge(self.nodes[parent_id], weight)
        child_edge = HdagEdge(self.nodes[child_id], weight)
        self.nodes[parent_id].add_child(child_edge)
        self.nodes[child_id].add_parent(parent_edge)

    def remove_edge(self, parent_id, child_id):
        """
        Remove a directed edge from a parent graph node to child graph node.

        :param parent_id: int
        :param child_id: int
        :return: None
        """
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        parent.remove_child(child_id)
        child.remove_parent(parent_id)

    def get_hdag_node(self, node_id):
        """
        Return the node associated with node_id from the graph.

        :param node_id: int
        :return: None
        """
        return self.nodes[node_id]

    def remove_non_maximal_inbound_edges(self):
        """
        Remove all non-maximal inbound edges from a graph node.  If the node has
        parents, only keep the parent with the greatest weighted edge.

        :return: None
        """
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
        """
        Partition the entire graph based on a weight threshold.

        :param weight_threshold: float
        :return: None
        """
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
        """
        Return the parentless nodes from the graph
        :return: None
        """
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
        """
        Add a parent to a node.  Helper function for Hdag class add_edge procedure.

        :param parent: int
        :return:
        """
        self.parents[parent.node.id] = parent

    def add_child(self, child):
        """
        Add a child to a node.  Help function for Hdag class add_edge procedure.

        :param child: int
        :return:
        """
        self.children[child.node.id] = child

    def remove_parent(self, parent_id):
        """
        Remove a parent from a node.  Helper function for Hdag class remove_edge procedure.

        :param parent_id: int
        :return:
        """
        self.parents.pop(parent_id)

    def remove_child(self, child_id):
        """
        Remove a child from a node.  Helper function for Hdag class remove_edge procedure.

        :param child_id:
        :return:
        """
        self.children.pop(child_id)


# class Partition:
#     def __init__(self, root, hierarchies):
#         self.id = root.root_id
#         self.root = root
#         self.children = hierarchies
#
#     def get_child_cells(self, child, patch):
#         return self.children[child].root.nodes_by_id[patch].patch.cells
#
#     def get_child_nodes_by_id(self, child):
#         return self.children[child].root.nodes_by_id
#
#     def get_root_patch_cells(self):
#         return self.root.root.patch.cells


def find_connected_hierarchies(patch_dict):
    """
    From a dictionary of all patches, find unique hierarchies that share patches.

    :param patch_dict: dictionary of patch objects
    :return: a list of tuples of connected hierarchies.
    """

    # initialize the list of contact patches
    contact_patches = set()

    # For every patch, find which hierarchies it belongs to, create a list of combinations,
    # attempt to add each hierachy pair to contact patches
    for patch in patch_dict.values():
        hierarchies = patch.hierarchies

        # Only try to find combinations for patches that are associated with more than
        # one hierarchy.
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


def calculate_depth(h1, h2, shared_patches):
    """
    Calculate node depth, level depth, and shared cell count for a pair of
    connected hierarchies.

    :param h1: hierarchy
    :param h2: hierarchy
    :param shared_patches: list of patch objects
    :return: depth statistics level_depth (float), node_depth(float), shared_cell_count (int)
    """
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
    """
    Assign weight an orientation to hierarchy nodes in the directed graph (Hdag object).

    :param h1: hierarchy
    :param h2: hierarchy
    :param weight: float
    :param h1_cc: int, the cell count of the hierarchy
    :param h2_cc: int, the cell count of the hierarchy
    :param hdag: the Hdag object to add edges to
    :return: None
    """
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

