# hdag.py
from treesegmentation import hierarchy, patch
from scipy.spatial import distance
from scipy.sparse import csc_matrix

import numpy as np


class HdagEdge:
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight


class HdagNode:
    def __init__(self, root):
        self.id = root.root_id
        self.root = root
        self.parents = {}
        self.children = {}


class Partition:
    def __init__(self, root, hierarchies):
        self.id = root.root_id
        self.root = root
        self.children = hierarchies


def find_connected_hierarchies(contact_patches):
    connected_hierarchies = {}
    for patch in contact_patches:
        patches = contact_patches[patch]
        p = np.array(list(patches))

        # Search for nodes that have more than one parent
        if len(p) > 1:
            # Create a cross product of p, find unique heirarchy pairs
            c = np.array(np.meshgrid(p, p)).T.reshape(-1, 2)
            # Check if a pair exists, if not add to dictionary of pairs
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


def add_edge(parent, child, weight, row, col, data):
    row.append(parent)
    col.append(child)
    data.append(weight)


def set_weight_and_orientation(h1, h2, weight, h1_cc, h2_cc, hdag):
    # IDs of the top patches for a hierachy.  Paradigm for the graph is
    # hdag_csg[parent][child] to show a directed edge from parent->child
    h1_id = h1.root_id
    h2_id = h2.root_id
    
    row = hdag[0]
    col = hdag[1]
    data = hdag[2]
    
    # See if we can assign edge direction based on height level first
    if h1.height > h2.height:
        #hdag_csg[h1_id][h2_id] = weight
        add_edge(h1_id, h2_id, weight, row, col, data)
        return
    elif h2.height > h1.height:
        #hdag_csg[h2_id][h1_id] = weight
        add_edge(h2_id, h1_id, weight, row, col, data)
        return

    # The following if/else trees are for determining edge direction
    # if the height level of the two hierarchies are equal.

    # Try to determine edge direction based on total hierarchy cell count
    if h1_cc > h2_cc:
        #hdag_csg[h1_id][h2_id] = weight
        add_edge(h1_id, h2_id, weight, row, col, data)
        return
    elif h2_cc > h1_cc:
        #hdag_csg[h2_id][h1_id] = weight
        add_edge(h2_id, h1_id, weight, row, col, data)
        return

    # Try to determine direction based on the cell count of the top patch
    # of each hierarchy
    if h1.root.patch.cell_count > h2.root.patch.cell_count:
        #hdag_csg[h1_id][h2_id] = weight
        add_edge(h1_id, h2_id, weight, row, col, data)
        return
    elif h2.root.patch.cell_count > h1.root.patch.cell_count:
        #hdag_csg[h2_id][h1_id] = weight
        add_edge(h2_id, h1_id, weight, row, col, data)
        return

    # Finally try to determine direction based on the location of each hierarchy's top patch 
    # centroid. First look for the "bottom most" centroid (highest 'i' value), then look for
    # the "right most" centroid (highest 'j' value).
    result = h1.root.patch.centroid - h2.root.patch.centroid
    
    if result[0] == 0:
        if result[1] > 0:
            #hdag_csg[h1_id][h2_id] = weight
            add_edge(h1_id, h2_id, weight, row, col, data)
        else:
            #hdag_csg[h2_id][h1_id] = weight
            add_edge(h2_id, h1_id, weight, row, col, data)
        return
    else:
        if result[0] > 0:
            #hdag_csg[h1_id][h2_id] = weight
            add_edge(h1_id, h2_id, weight, row, col, data)
        else:
            #hdag_csg[h2_id][h1_id] = weight
            add_edge(h2_id, h1_id, weight, row, col, data)
        return


def calculate_edge_weight(hierarchies, connected_hierarchies, weights):
    row = []
    col = []
    data = []
    
    hdag = [row, col, data]

    hierarchy_dict = {}
    hdag_nodes = {}
    for hierarchy in hierarchies:
        hierarchy_dict[hierarchy.root_id] = hierarchy

    for i, j in connected_hierarchies:
        h1 = hierarchy_dict[i]
        h2 = hierarchy_dict[j]

        hdag_nodes[i] = HdagNode(hierarchy_dict[i])
        hdag_nodes[j] = HdagNode(hierarchy_dict[j])

        shared_patches = np.intersect1d(h1.nodes_by_id_array, h2.nodes_by_id_array)
        level_depth, node_depth, shared_cell_count = calculate_depth(h1, h2, shared_patches)

        # Get height adjusted centroid for each hiearchy and total cell counts
        h1_hac, h1_cell_count = calculate_hac(h1)
        h2_hac, h2_cell_count = calculate_hac(h2)
        
        shared_ratio = shared_cell_count / (h1_cell_count + h2_cell_count - shared_cell_count)

        # Get the top patch centroids
        h1_top = h1.root.patch.centroid
        h2_top = h2.root.patch.centroid

        # Calculate top and centroid distance statistic.  Reshape top patch 
        # centroids as 2-d numpy arrays because cdist() expects a 2-d array
        top_distance = 1 / (distance.cdist(np.reshape(h1_top, (-1, 2)), 
            np.reshape(h2_top, (-1, 2)), 'euclidean')[0][0])

        centroid_distance = 1 / (distance.cdist(np.reshape(h1_hac, (-1, 2)), 
            np.reshape(h2_hac, (-1, 2)), 'euclidean')[0][0])
        
        # Convert the statistics to a numpy float array
        scores = np.array([level_depth, node_depth, shared_ratio, top_distance, centroid_distance],
            dtype=np.float32)

        # Finally caculate the edge weight for h1, h2.  Find the direction of the weighted edge.
        edge_weight = np.sum((scores * weights)) 
        #set_weight_and_orientation(h1, h2, edge_weight, h1_cell_count, h2_cell_count, hdag_csg)
        set_weight_and_orientation(h1, h2, edge_weight, h1_cell_count, h2_cell_count, hdag)
        
    #return hdag_csg
    return csc_matrix((data, (row, col)), dtype=np.float16), hdag_nodes


def partition_graph(HDAG, hdag_nodes, weight_threshold):
    HDAG_coo = HDAG.tocoo()
    data = HDAG_coo.data
    row = HDAG_coo.row
    col = HDAG_coo.col

    source_nodes = []

    # Create edges based on HDAG coo representaiton
    for i in range(len(data)):
        parent = hdag_nodes[row[i]]
        child = hdag_nodes[col[i]]
        weight = data[i]
        parent_edge_to_add = HdagEdge(child, weight)
        child_edge_to_add = HdagEdge(parent, weight)
        parent.children[child.id] = parent_edge_to_add
        child.parents[parent.id] = child_edge_to_add

    # Partition based on non-maximal inbound edges
    for node in hdag_nodes:
        hdag_node = hdag_nodes[node]
        max_parent_weight = -1
        max_parent_key = -1

        # Choose the edges to remove
        non_maximal_inbound_edges = []
        for parent in hdag_node.parents:
            parent_edge = hdag_node.parents[parent]
            weight = parent_edge.weight
            if weight > max_parent_weight:
                non_maximal_inbound_edges.append(max_parent_key)
                max_parent_weight = weight
                max_parent_key = parent
            else:
                non_maximal_inbound_edges.append(parent)

        # remove every edge except the maximum parent
        for i in range(len(non_maximal_inbound_edges)):
            edge_to_remove = non_maximal_inbound_edges[i]
            if edge_to_remove != -1:
                hdag_node.parents.pop(edge_to_remove)
                hdag_nodes[edge_to_remove].children.pop(node)

    # Now partition based on weight threshold
    for node in hdag_nodes:
        hdag_node = hdag_nodes[node]
        edge_to_remove = -1
        for parent in hdag_node.parents:
            parent_edge = hdag_node.parents[parent]
            weight = parent_edge.weight
            if weight < weight_threshold:
                edge_to_remove = parent

        if edge_to_remove != -1:
            hdag_node.parents.pop(edge_to_remove)
            hdag_nodes[edge_to_remove].children.pop(node)

    # get parentless source nodes, these are the trees
    sources = []
    for node in hdag_nodes:
        hdag_node = hdag_nodes[node]
        if len(hdag_node.parents) == 0:  # No parents, source node
            sources.append(node)

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

    return source_nodes






