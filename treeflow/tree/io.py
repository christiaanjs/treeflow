import ete3
import numpy as np
from treeflow import DEFAULT_FLOAT_DTYPE_NP
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.taxon_set import DictTaxonSet

_EPSILON = 1e-6


def remove_zero_edges(
    tree: NumpyRootedTree, epsilon: float = _EPSILON
) -> NumpyRootedTree:
    heights = np.array(tree.heights)
    child_indices = tree.topology.child_indices
    for node_index in tree.topology.postorder_node_indices:
        child_heights = heights[child_indices[node_index]]
        heights[node_index] = np.max(
            (heights[node_index], np.max(child_heights) + epsilon)
        )
    return NumpyRootedTree(heights=heights, topology=tree.topology)


_remove_zero_edges_func = remove_zero_edges


def parse_newick(
    newick_file: str, remove_zero_edges: bool = True, epsilon: float = _EPSILON
) -> NumpyRootedTree:
    """Return leaves followed by nodes (postorder)"""
    t = ete3.Tree(newick_file)
    ordered_nodes = sorted(t.traverse("postorder"), key=lambda n: not n.is_leaf())

    indices = {n: i for i, n in enumerate(ordered_nodes)}
    parent_indices = np.array([indices[n.up] for n in ordered_nodes[:-1]])

    root_distances = np.array(
        [t.get_distance(n) for n in ordered_nodes], dtype=DEFAULT_FLOAT_DTYPE_NP
    )  # TODO: Optimise
    root_height = max(root_distances)
    heights = root_height - root_distances

    taxon_count = (len(ordered_nodes) + 1) // 2
    taxon_set = DictTaxonSet([x.name for x in ordered_nodes[:taxon_count]])
    tree = NumpyRootedTree(
        heights=heights, parent_indices=parent_indices, taxon_set=taxon_set
    )
    if remove_zero_edges:
        tree = _remove_zero_edges_func(tree, epsilon=epsilon)

    return tree


__all__ = [parse_newick.__name__, remove_zero_edges.__name__]
