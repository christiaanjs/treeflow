import libsbn
import treeflow.tree_processing
import numpy as np


def get_instance(newick_file, dated=True, name="treeflow"):
    inst = libsbn.rooted_instance(name)
    inst.read_newick_file(newick_file)
    if dated:
        inst.parse_dates_from_taxon_names(True)
    else:
        inst.set_dates_to_be_constant(True)
    return inst


def get_tree_info(inst):
    tree = inst.tree_collection.trees[0]
    parent_indices = np.array(tree.parent_id_vector())
    node_heights = np.array(tree.node_heights)
    node_bounds = np.array(tree.node_bounds)
    taxon_count = (parent_indices.shape[0] + 2) // 2
    return treeflow.tree_processing.TreeInfo(
        dict(heights=node_heights, topology=dict(parent_indices=parent_indices)),
        node_bounds[taxon_count:],
    )
