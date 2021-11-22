import ete3
import numpy as np
from treeflow import DEFAULT_FLOAT_DTYPE_NP
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.taxon_set import DictTaxonSet


def parse_newick(newick_file: str) -> NumpyRootedTree:
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
    return NumpyRootedTree(
        heights=heights, parent_indices=parent_indices, taxon_set=taxon_set
    )


__all__ = [parse_newick.__name__]
