from treeflow.distributions.tree.birthdeath.birth_death_contemporary_sampling import (
    BirthDeathContemporarySampling,
)
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF


def test_BirthDeathContemporarySampling_log_prob():
    newick = "((((human:0.024003,(chimp:0.010772,bonobo:0.010772):0.013231):0.012035,gorilla:0.036038):0.033087000000000005,orangutan:0.069125):0.030456999999999998,siamang:0.099582);"
    tree = convert_tree_to_tensor(parse_newick(newick))
    birth_diff_rate = tf.constant(1.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    relative_death_rate = tf.constant(0.5, dtype=DEFAULT_FLOAT_DTYPE_TF)
    expected = 1.2661341104158121  # From BEAST 2 BirthDeathGerhard08ModelTest

    dist = BirthDeathContemporarySampling(
        tree.taxon_count, birth_diff_rate, relative_death_rate
    )
    res = dist.log_prob(tree)

    assert_allclose(res.numpy(), expected)


def test_BirthDeathContemporarySampling_log_prob_vec(hello_newick_file):
    tree = convert_tree_to_tensor(parse_newick(hello_newick_file))
    tree_b = tree.with_node_heights(tf.expand_dims(tree.node_heights, 0))
    birth_diff_rate = tf.constant([1.0, 1.2], dtype=DEFAULT_FLOAT_DTYPE_TF)
    relative_death_rate = tf.constant([0.5, 0.3], dtype=DEFAULT_FLOAT_DTYPE_TF)

    dist = BirthDeathContemporarySampling(
        tree.taxon_count, birth_diff_rate, relative_death_rate
    )
    res = dist.log_prob(tree_b)
    assert res.numpy().shape == birth_diff_rate.numpy().shape
