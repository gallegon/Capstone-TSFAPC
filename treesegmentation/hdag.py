# hdag.py
from treesegmentation import hierarchy, patch
from scipy.spatial import distance

import numpy as np

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
                if (i,j) in connected_hierarchies or (j,i) in connected_hierarchies:
                    continue
                else:
                    if i != j:
                        connected_hierarchies[(i,j)] = (i,j)

    return connected_hierarchies

def calculate_depth(h1, h2, shared_patches, h_csg):
    shared_cell_count = 0

    level_depth = np.inf
    node_depth = np.inf

    for contact_patch in shared_patches:
        # Get level depth
        level_depth = min(level_depth, h1.height - h1.nodes_by_id[contact_patch].patch.height_level, 
            h2.height - h2.nodes_by_id[contact_patch].patch.height_level)
        
        # Get node depth from sparse graph
        node_depth = min(node_depth, h_csg[h1.root_id][contact_patch],
            h_csg[h2.root_id][contact_patch])
        
        shared_cell_count += h1.nodes_by_id[contact_patch].patch.cell_count  

    return 1 / level_depth, 1 / node_depth, shared_cell_count

def calculate_hac(h):
    hierarchy_cell_count = 0

    hac_numerator = np.array([0, 0], dtype=np.float64)
    hac_denominator = np.array([0, 0], dtype=np.float64)

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


def set_weight_and_orientation(h1, h2, weight, h1_cc, h2_cc, hdag_csg):
    # IDs of the top patches for a hierachy.  Paradigm for the graph is
    # hdag_csg[parent][child] to show a directed edge from parent->child
    h1_id = h1.root_id
    h2_id = h2.root_id
    
    # See if we can assign edge direction based on height level first
    if h1.height > h2.height:
        hdag_csg[h1_id][h2_id] = weight
        return
    else:
        hdag_csg[h2_id][h1_id] = weight
        return

    # The following if/else trees are for determining edge direction
    # if the height level of the two hierarchies are equal.

    # Try to determine edge direction based on total hierarchy cell count
    if h1_cc > h2_cc:
        hdag_csg[h1_id][h2_id] = weight
        return
    else:
        hdag_csg[h2_id][h1_id] = weight
        return

    # Try to determine direction based on the cell count of the top patch
    # of each hierarchy
    if h1.root.patch.cell_count > h2.root.patch.cell_count:
        hdag_csg[h1_id][h2_id] = weight
        return
    else:
        hdag_csg[h2_id][h1_id] = weight
        return

    # Finally try to determine direction based on the location of each
    # Hierarchy's top patch centroid.
    result = h1.root.patch.centroid - h2.root.patch.centroid
    
    if result[0] == 0:
        if result[1] > 0:
            hdag_csg[h1_id][h2_id] = weight
        else:
            hdag_csg[h2_id][h1_id] = weight
    else:
        if result[0] > 0:
            hdag_csg[h1_id][h2_id] = weight
        else:
            hdag_csg[h2_id][h1_id] = weight


def calculate_edge_weight(hierarchies, connected_hierarchies, h_csg, weights):
    size = len(h_csg[0])
    hdag_csg = np.zeros((size, size))

    hierarchy_dict = {}
    for hierarchy in hierarchies:
        hierarchy_dict[hierarchy.root_id] = hierarchy

    for i, j in connected_hierarchies:
        h1 = hierarchy_dict[i]
        h2 = hierarchy_dict[j]

        shared_patches = np.intersect1d(h1.nodes_by_id_array, h2.nodes_by_id_array)
        level_depth, node_depth, shared_cell_count = calculate_depth(h1, h2, shared_patches, h_csg)

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
            dtype=np.float64)

        # Finally caculate the edge weight for h1, h2
        edge_weight = np.sum((scores * weights)) 

        set_weight_and_orientation(h1, h2, edge_weight, h1_cell_count, h2_cell_count, hdag_csg)
        
        """
        print(f"**********")
        print(f"h1 cell count {h1_cell_count}")
        print(f"h2 cell count {h2_cell_count}")
        print(f"shared 1: {shared_cell_count}")
        print(f"h1 top centroid {h1_top}")
        print(f"h1 HAC {h1_hac}")
        print(f"level depth:{level_depth}")
        print(f"node depth: {node_depth}")
        print(f"shared ratio: {shared_ratio}")
        print(f"top distance: {top_distance}")
        print(f"centroid distance: {centroid_distance}")
        print(f"edge weight: {edge_weight}")
        """
    return hdag_csg

