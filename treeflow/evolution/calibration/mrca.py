import typing as tp
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


def get_common_ancestors(
    topology: NumpyTreeTopology, indices: tp.Iterable[int]
) -> tp.Set[int]:
    ancestors: tp.Set[int] = set()
    node_count = topology.node_count
    for base_index in indices:
        ancestors_remaining = True
        index = base_index
        while ancestors_remaining and index < (node_count - 1):
            parent = topology.parent_indices[index]
            if parent in ancestors:
                ancestors_remaining = False
                ancestors = set([x for x in ancestors if x >= parent])
            else:
                ancestors.add(parent)
                index = parent
    return ancestors


def get_mrca_index(topology: NumpyTreeTopology, taxa: tp.Iterable[str]) -> int:
    assert topology.taxon_set is not None
    all_taxa = list(topology.taxon_set)
    indices = [all_taxa.index(taxon) for taxon in taxa]
    ancestors = get_common_ancestors(topology, indices)
    return min(ancestors)
