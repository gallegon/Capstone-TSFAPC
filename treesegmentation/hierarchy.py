import numpy as np

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import shortest_path

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


def compute_hierarchies(all_patches):
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
    contact_patches = {}
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
        hierarchies.append(hierarchy)

        # Create a list of contact patches
        for node in reachable_nodes_by_id:
            contact_patches.setdefault(node, set()).add(root.patch_id)

    return hierarchies, contact_patches


def hierarchy_as_raster(labeled_grid, hierarchy):
    hierarchy_grid = np.full(labeled_grid.shape, 255)
    for node in hierarchy.nodes_by_id.values():
        for x, y in node.patch.cells:
            patch_id = labeled_grid[x, y]
            patch_height = hierarchy.nodes_by_id[patch_id].patch.height_level
            hierarchy_grid[x, y] = 255 * (1 - patch_height / hierarchy.height)
    return hierarchy_grid


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