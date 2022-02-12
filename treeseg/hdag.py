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
    for patch in contact_patches:
        patches = contact_patches[patch]
        p = list(patches)
        # Search for nodes that have more than one parent
        if len(p) > 1:
            # Create a cross product of p, find unique heirarchy pairs
            c = itertools.combinations(p, 2)
            # Check if a pair exists, if not add to dictionary of pairs
            '''
            TODO: clean this up, maybe create a list of tuples instead of a dictionary
            see how we can reduce computations here
            '''
            for i, j in c:
                if (i, j) in connected_hierarchies or (j, i) in connected_hierarchies:
                    continue
                else:
                    if i != j:
                        connected_hierarchies[(i, j)] = (i, j)

    return connected_hierarchies


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


'''
TODO: Finalize with Sam, I moved this to the hierarchy class so we don't have to
Recompute a HAC every time, it seems to have sped up the code significantly since
We only have to compute each HAC and cell count once now...
'''
def calculate_hac(h):
    # "Running total" for cell count, weighted centroids
    hierarchy_cell_count = 0

    hac_numerator = np.array([0, 0], dtype=np.float64)
    hac_denominator = np.array([0, 0], dtype=np.float64)

    # For every node in hierarchy, do the centroid weighting and cell counts
    for node in h.nodes_by_id:
        patch_cell_count = h.nodes_by_id[node].patch.cell_count
        patch_level = h.nodes_by_id[node].patch.height_level
        dh = h.height - patch_level

        patch_centroid = h.nodes_by_id[node].patch.centroid

        hierarchy_cell_count += patch_cell_count
        
        hac_constant = patch_cell_count * (dh + 1)
        hac_constant_np = np.array([hac_constant, hac_constant])

        hac_numerator += hac_constant_np * patch_centroid
        hac_denominator += hac_constant_np

    height_adjusted_centroid = hac_numerator / hac_denominator

    return height_adjusted_centroid, hierarchy_cell_count


def set_weight_and_orientation(h1, h2, weight, h1_cc, h2_cc, hdag):
    # IDs of the top patches for a hierachy.  Paradigm for the graph is
    # hdag_csg[parent][child] to show a directed edge from parent->child
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

        # Get height adjusted centroid for each hierarchy and total cell counts
        #h1_hac, h1_cell_count = calculate_hac(h1)
        #h2_hac, h2_cell_count = calculate_hac(h2)
        
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
        source_nodes.append(Partition(hdag_nodes[source].root, hierarchies))

    # Return an array of the partitions
    return source_nodes


def partitions_to_labeled_grid(partitions, x, y):
    labeled_grid = np.zeros(shape=(x,y), dtype=np.int64)
    for partition in partitions:
        partition_cells = []
        #print(partition.__dict__)
        partition_cells.append(partition.root.root.patch.cells)
        for child in partition.children:
            for patch in partition.children[child].root.nodes_by_id:
                #print(partition.children[child].root.nodes_by_id[patch].patch.__dict__)
                partition_cells.append(partition.children[child].root.nodes_by_id[patch].patch.cells)
        for cells in partition_cells:
            for x, y in cells:
                labeled_grid[x][y] = partition.id
    #print(labeled_grid)
    gaussian_filter(labeled_grid, sigma=1)
    return labeled_grid




