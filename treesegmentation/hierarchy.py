"""Module for dealing with the Hierarchy data object.
"""

import numpy as np

# For new node_depth implementation
from collections import deque


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
    def __init__(self, root, nodes_by_id, node_depths_by_id):
        self.root_id = root.patch_id
        self.height = root.patch.height_level
        self.root = root
        self.nodes_by_id = nodes_by_id
        self.nodes_by_id_array = np.array(list(nodes_by_id.keys()))
        self.node_depths_by_id = node_depths_by_id
        self.height_adjusted_centroid = None
        self.cell_count = 0

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


def compute_hierarchies(all_patches, patch_dict):
    """
    Compute the directed hierarchy graph from the Patch objects.  This sets up
    a directed graph based on a given Patch's neighbors.  Does a traversal of
    each local maxima in the graph (parentless nodes) and sets up hierarchies
    as specified in the algorithm.

    :param all_patches: The list of Patch objects from previous step of the algorithm
    :param patch_dict: The dictionary of Patch objects
    :return: a list of Hierarchy objects
    """
    # Basic pipeline of what's happening:
    # - Transform every Patch into a HierarchyNode.
    # - Find the roots (nodes with no parents).
    # - Compute and return a Hierarchy for each root node.

    # Compute nodes (and roots)
    all_patches_by_id = {patch.id: patch for patch in all_patches}
    all_nodes_by_id = {}
    root_nodes = []

    for patch in all_patches:
        parents, children = set(), set()
        for neighbor_id in patch.neighboring_patches:
            neighbor = all_patches_by_id[neighbor_id]
            if neighbor.height_level < patch.height_level:
                children.add(neighbor.id)
            elif neighbor.height_level > patch.height_level:
                parents.add(neighbor.id)
        node = HierarchyNode(patch.id, patch, parents, children)
        if len(parents) == 0:
            root_nodes.append(node)
        all_nodes_by_id[node.patch_id] = node

    hierarchies = []

    # Initialize a stack for computing node depth while the hierarchies are being created
    depth_stack = deque()

    for root in root_nodes:
        # For each root, compute all reachable nodes.
        reachable_nodes_by_id = {}
        node_depths_by_id = {}
        root_height = root.patch.height_level
        queued_nodes = {root}
        depth_stack.append(0)

        while len(queued_nodes) != 0:
            next_queue = set()
            # Get the current node depth
            node_depth = depth_stack.pop()
            for node in queued_nodes:
                reachable_nodes_by_id[node.patch_id] = node

                # Keep track of depth statistics
                level_depth = root_height - node.patch.height_level
                node_depths_by_id[node.patch_id] = (level_depth, node_depth)

                for child_id in node.children:
                    next_queue.add(all_nodes_by_id[child_id])
                    # For the children of the node, increment node_depth by 1
                    depth_stack.append(node_depth + 1)
            queued_nodes = set(next_queue)

        hierarchy = Hierarchy(root, reachable_nodes_by_id, node_depths_by_id)

        # Calculate and set the height-adjusted centroid and cell_count for this
        # hierarchy
        hac, cell_count = calculate_hac(hierarchy)
        hierarchy.height_adjusted_centroid = hac
        hierarchy.cell_count = cell_count

        # Try to add the hierarchy to an associated patch
        for node in reachable_nodes_by_id:
            patch = patch_dict[node]
            patch.add_hierarchy(hierarchy)

        hierarchies.append(hierarchy)


    print("computed hierarchies")

    return hierarchies


def calculate_hac(h):
    """
    Computes the height-adjusted centroid for each hierarchy. Follows the
    procedure described by the tree segmentation algorithm.  This is used
    in the weighted graph creation to assign edge weights.

    :param h: Hierarchy to calculate height-adjusted centroid
    :return: The HAC and cell count of the hierarchy
    """
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
