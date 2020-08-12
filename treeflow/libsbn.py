import libsbn
import treeflow.tree_processing
import numpy as np

def get_instance(newick_file, name='treeflow'):
    inst = libsbn.rooted_instance(name)
    inst.read_newick_file(newick_file)
    return inst

def get_tree_info(inst):
    tree = inst.tree_collection.trees[0]
    parent_indices = np.array(tree.parent_id_vector())
    node_heights = np.array(tree.node_heights)
    node_bounds = np.array(tree.node_bounds)
    taxon_count = (parent_indices.shape[0] + 2) // 2
    return treeflow.tree_processing.TreeInfo(
        dict(
            heights=np.concatenate([node_bounds[:taxon_count], node_heights]),
            topology=dict(parent_indices=parent_indices)
        ),
        node_bounds    
    )