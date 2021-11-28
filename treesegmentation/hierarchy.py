# hierarchy.py
import numpy as np

from .las2img import *
import treesegmentation.patch as patch


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
    
    # Compute Hierarchy for each root node
    hierarchies = []
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
    
    return hierarchies


def hierarchy_as_raster(labeled_grid, hierarchy):
    hierarchy_grid = np.full(labeled_grid.shape, 255)
    for node in hierarchy.nodes_by_id.values():
        for x, y in node.patch.cells:
            patch_id = labeled_grid[x, y]
            patch_height = hierarchy.nodes_by_id[patch_id].patch.height_level
            hierarchy_grid[x, y] = 255 * (1 - patch_height / hierarchy.height)
    return hierarchy_grid
