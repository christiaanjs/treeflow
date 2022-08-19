import numpy as np
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.birthdeath.yule import Yule
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from numpy.testing import assert_allclose


def test_yule():
    birth_rate = tf.constant(10.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    numpy_tree = parse_newick("((A:1.0,B:1.0):1.0,C:2.0);")
    tree = convert_tree_to_tensor(numpy_tree)
    dist = Yule(tree.taxon_count, birth_rate)
    res = dist.log_prob(tree)
    expected = calc_alt_yule_log_p(numpy_tree, birth_rate.numpy())
    assert_allclose(res.numpy(), expected)


# From BEAST 2: test.beast.evolution.speciation.YuleModelTest
def calc_alt_yule_log_p(tree: NumpyRootedTree, birth_rate):
    n = tree.taxon_count
    log_p = (n - 1) * np.log(birth_rate) - birth_rate * tree.node_heights[-1]
    for height in tree.node_heights:
        log_p -= birth_rate * height
    return log_p
