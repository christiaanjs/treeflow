import numpy as np
import typing as tp
import bito
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


def get_instance(newick_file, dated=True, name="treeflow"):

    inst = bito.rooted_instance(name)
    inst.read_newick_file(newick_file)
    if dated:
        inst.parse_dates_from_taxon_names(True)
    else:
        inst.set_dates_to_be_constant(True)
    return inst


def get_tree_info(inst) -> tp.Tuple[NumpyRootedTree, np.ndarray]:
    bito_tree = inst.tree_collection.trees[0]
    parent_indices = np.array(bito_tree.parent_id_vector())
    node_heights = np.array(bito_tree.node_heights)
    tree = NumpyRootedTree(
            heights=node_heights,
            topology=NumpyTreeTopology(parent_indices=parent_indices),
        )
    
    node_bounds = np.array(bito_tree.node_bounds)[tree.taxon_count:]
    return (
        tree,
        node_bounds,
    )
