# hierarchy.py

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class HierarchyNode:
    def __init__(self, patch_id, patch, parents, children):
        self.patch_id = patch_id
        self.patch = patch
        self.parents = parents
        self.children = children

    def __str__(self):
        lines = [
            f"HNode#{self.patch_id}:",
            f"  height = {self.patch.height_level}",
            f"  parents[{len(self.parents)} = [",
            *",\n".join([f"    {parent!r}" for parent in self.parents]).splitlines(),
            "  ]"
            f"  children[{len(self.children)} = [",
            *",\n".join([f"    {child!r}" for child in self.children]).splitlines(),
            "  ]"
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"HNode<{self.patch.height_level}" \
               f", parents[{len(self.parents)}], children[{len(self.children)}], #{self.patch_id}>"


class Hierarchy:
    def __init__(self, root, nodes_by_id):
        self.root_id = root.patch_id
        self.height = root.patch.height_level
        self.root = root
        self.nodes_by_id = nodes_by_id
        self.nodes_by_id_array = np.array(list(nodes_by_id.keys()))

    def __str__(self):
        lines = [
            f"Hierarchy#{self.root_id}:",
            f"  height = {self.height}",
            f"  nodes[{len(self.nodes_by_id)}] = [",
            *",\n".join([f"    {node!r}" for node in self.nodes_by_id.values()]).splitlines(),
            "  ]"
        ]
        return "\n".join(lines)

    def __repr__(self):
        return f"Hierarchy<{self.height}, {len(self.nodes_by_id)}, #{self.root_id}>"


def compute_hierarchies(all_patches):
    # Basic pipeline of what's happening:
    # - Transform every Patch into a HierarchyNode.
    # - Find the roots (nodes with no parents).
    # - Compute and return a Hierarchy for each root node.

    # Compute nodes (and roots)
    all_patches_by_id = {patch.id: patch for patch in all_patches}
    all_nodes_by_id = {}
    root_nodes = []

    # Initialize a sparse matrix
    hierarchy_csg = initialize_hierarchy_csg(len(all_patches))

    for patch in all_patches:
        parents, children = set(), set()
        for neighbor_id in patch.neighboring_patches:
            neighbor = all_patches_by_id[neighbor_id]
            if neighbor.height_level < patch.height_level:
                hierarchy_csg[patch.id][neighbor_id] = 1
                children.add(neighbor.id)
            elif neighbor.height_level > patch.height_level:
                hierarchy_csg[neighbor_id][patch.id] = 1
                parents.add(neighbor.id)
        node = HierarchyNode(patch.id, patch, parents, children)
        if len(parents) == 0:
            root_nodes.append(node)
        all_nodes_by_id[node.patch_id] = node
        #print(f"Patch {patch.id}: {children}")

    # Compute Hierarchy for each root node
    hierarchies = []
    contact_patches = {}

    for root in root_nodes:
        # For each root, compute all reachable nodes.
        reachable_nodes_by_id = {}
        queued_nodes = {root}
        while len(queued_nodes) != 0:
            next_queue = set()
            for node in queued_nodes:
                reachable_nodes_by_id[node.patch_id] = node
                for child_id in node.children:
                    next_queue.add(all_nodes_by_id[child_id])
            queued_nodes = set(next_queue)
        hierarchy = Hierarchy(root, reachable_nodes_by_id)
        hierarchies.append(hierarchy)

        # Create a list of contact patches
        for node in reachable_nodes_by_id:
            contact_patches.setdefault(node, set()).add(root.patch_id)
            # TODO: Remove this commented out line, keep for testing
            #print(f"Root node {root.patch_id}: {node}")
    
    # TODO: Remove this after completion
    '''
    for patch in contact_patches:
        l = len(contact_patches[patch])
        t = type(contact_patches[patch])
        print(f"patch {patch}:{contact_patches[patch]} [len {l}] [type {t}]")
    '''
    # Create a distance matrix from one node to another.
    node_dist_matrix, predecessors = dijkstra(
        csgraph=hierarchy_csg, directed=True, return_predecessors=True)
    
    #print(np.matrix(dist_matrix))
    return hierarchies, node_dist_matrix, contact_patches


def hierarchy_as_raster(labeled_grid, hierarchy):
    hierarchy_grid = np.full(labeled_grid.shape, 255)
    for node in hierarchy.nodes_by_id.values():
        for x, y in node.patch.cells:
            patch_id = labeled_grid[x, y]
            patch_height = hierarchy.nodes_by_id[patch_id].patch.height_level
            hierarchy_grid[x, y] = 255 * (1 - patch_height / hierarchy.height)
    return hierarchy_grid


def initialize_hierarchy_csg(size):
    # Initialize hierachies as a sparse matrix
    size += 1
    hierarchy_csg = np.zeros((size, size))
    np.fill_diagonal(hierarchy_csg, 1)
    return hierarchy_csg
