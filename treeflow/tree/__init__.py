"""
Classes for representing trees in Tensorflow

We use `attr` based classes as they are supported by `tf.nest`.
Custom behaviour can be added subclassing `attr` classes as long as
support for the standard constructor argument order is preserved.
"""

from treeflow.tree.taxon_set import DictTaxonSet, TupleTaxonSet
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
    tree_from_arrays,
)
from treeflow.tree.io import parse_newick, write_tensor_trees

__all__ = ["parse_newick", "convert_tree_to_tensor", "write_tensor_trees"]
