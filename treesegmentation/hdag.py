# hdag.py
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

import numpy as np
import itertools


class Hdag:
    def __init__(self):
        self.nodes = {}

    # Initialize the graph from an array of hierarchies
    def initialize_from_hierarchies(self, hierarchies):
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


def find_connected_hierarchies(contact_patches):
    connected_hierarchies = {}

    # Look for through the contact patches
    for patches in contact_patches.values():
        p = [h.root_id for h in patches]
        # Create a cross product of p, find unique heirarchy pairs
        patch_combinations = itertools.combinations(p, 2)

        for i, j in patch_combinations:
            # Check if a pair exists, if not add to dictionary of pairs
            if (i, j) in connected_hierarchies or (j, i) in connected_hierarchies:
                continue
            else:
                if i != j:
                    connected_hierarchies[(i, j)] = (i, j)

    return connected_hierarchies


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


def partition_graph(HDAG, weight_threshold):
    source_nodes = []

    # map of hierarchies to partitions
    hp_map = {}

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
        partition = Partition(hdag_nodes[source].root, hierarchies)
        source_nodes.append(partition)

        # create a map of hierarchies and partitions
        for hierarchy in hierarchies:
            hp_map[hierarchy] = partition.id

    # Return an array of the partitions
    return source_nodes, hp_map


def partitions_to_labeled_grid(partitions, x, y):
    labeled_grid = np.zeros(shape=(x,y), dtype=np.int64)
    for partition in partitions:
        partition_cells = []
        #print(partition.__dict__)
        partition_cells.append(partition.root.root.patch.cells)
        for child in partition.children:
            for patch in partition.children[child].root.nodes_by_id:
                partition_cells.append(partition.children[child].root.nodes_by_id[patch].patch.cells)
        for cells in partition_cells:
            for x, y in cells:
                labeled_grid[x][y] = partition.id

    return labeled_grid


def adjust_partitions(patch_dict, labeled_partitions, contact, hp_map):
    adjusted_partitions = labeled_partitions.copy()

    #patch_dict = patches_to_dict(all_patches)

    for contact_patch_id, associated_hierarchies in contact.items():
        #patch = all_patches[contact_patch_id - 1]
        patch = patch_dict[contact_patch_id]
        patch_cells = patch.cells
        centroid = patch.centroid

        min_distance = np.inf
        closest_hierarchy = None

        # Get the centroids for each hierarchy that claims the contact patch
        for hierarchy in associated_hierarchies:
            centroid_distance = (distance.cdist(np.reshape(centroid, (-1, 2)),
                                                np.reshape(hierarchy.height_adjusted_centroid, (-1, 2)),
                                                'euclidean')[0][0])

            if centroid_distance < min_distance:
                closest_hierarchy = hierarchy
                min_distance = centroid_distance

        partition_to_adjust = hp_map[closest_hierarchy.root_id]

        for i, j in patch_cells:
            adjusted_partitions[i][j] = partition_to_adjust

    return adjusted_partitions
