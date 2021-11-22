import numpy as np
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree


def get_anchor_heights(tree: NumpyRootedTree) -> np.ndarray:
    taxon_count = tree.taxon_count
    anchor_heights = np.zeros_like(tree.heights)
    anchor_heights[..., :taxon_count] = tree.heights[..., :taxon_count]

    for i in tree.topology.postorder_node_indices:
        anchor_heights[..., i] = np.max(
            anchor_heights[..., tree.topology.child_indices[i]], axis=-1
        )

    return anchor_heights[..., taxon_count:]
