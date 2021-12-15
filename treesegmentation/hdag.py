# hdag.py
from treesegmentation import hierarchy, patch
from scipy.spatial import distance

import numpy as np

def find_connected_hierarchies(contact_patches):
    connected_hierarchies = {}
    for patch in contact_patches:
        patches = contact_patches[patch]
        p = np.array(list(patches))

        if len(p) > 1:
            c = np.array(np.meshgrid(p, p)).T.reshape(-1, 2)
            for i, j in c:
                if (i,j) in connected_hierarchies or (j,i) in connected_hierarchies:
                    continue
                else:
                    if i != j:
                        connected_hierarchies[(i,j)] = (i,j)

    return connected_hierarchies
        
def level_depth(hierarchies, connected_hierarchies, h_csg):
    hierarchy_dict = {}
    
    for hierarchy in hierarchies:
        hierarchy_dict[hierarchy.root_id] = hierarchy

    for i, j in connected_hierarchies:
        h1 = hierarchy_dict[i]
        h2 = hierarchy_dict[j]

        shared_patches = np.intersect1d(h1.nodes_by_id_array, h2.nodes_by_id_array)
        print(f"Shared patches by {h1.root_id} {h2.root_id}: {shared_patches}")
        
        level_depth = np.inf
        node_depth = np.inf

        for contact_patch in shared_patches:
            level_depth = min(level_depth, h1.root.patch.height_level - h1.nodes_by_id[contact_patch].patch.height_level)
            level_depth = min(level_depth, h2.root.patch.height_level - h2.nodes_by_id[contact_patch].patch.height_level)
            node_depth = min(node_depth, h_csg[h1.root_id][contact_patch])
            node_depth = min(node_depth, h_csg[h2.root_id][contact_patch])

        h1_cell_count = 0
        h2_cell_count = 0
        shared_cell_count = 0
        
        for node in h1.nodes_by_id:
            h1_cell_count += h1.nodes_by_id[node].patch.cell_count
        
        for node in h2.nodes_by_id:
            h2_cell_count += h2.nodes_by_id[node].patch.cell_count

        for node in h1.nodes_by_id:
            shared_cell_count += h1.nodes_by_id[node].patch.cell_count

        shared_ratio = shared_cell_count / (h1_cell_count + h2_cell_count - shared_cell_count)

        # TODO: Clean this up!
        arr1 = []
        arr2 = []
        h1_top_centroid = h1.root.patch.centroid
        h2_top_centroid = h2.root.patch.centroid
        arr1.append(h1_top_centroid)
        arr2.append(h2_top_centroid)

        top_distance = 1 / (distance.cdist(arr1, arr2, 'euclidean'))
        print(f"h1 cell count {h1_cell_count}")
        print(f"h2 cell count {h2_cell_count}")
        print(f"level depth:{level_depth}")
        print(f"node depth: {node_depth}")
        print(f"shared ratio: {shared_ratio}")
        print(f"top distance: {top_distance}")



