import tensorflow as tf
from treeflow.birthdeath import BirthDeath
import treeflow.tree_processing
from numpy.testing import assert_allclose
from treeflow import DEFAULT_FLOAT_DTYPE_TF

def test_birth_death():
    newick = '((((human:0.024003,(chimp:0.010772,bonobo:0.010772):0.013231):0.012035,gorilla:0.036038):0.033087000000000005,orangutan:0.069125):0.030456999999999998,siamang:0.099582);'
    tree = treeflow.tree_processing.tree_to_tensor(treeflow.tree_processing.parse_newick(newick)[0])
    birth_diff_rate = tf.convert_to_tensor(1.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    relative_death_rate = tf.convert_to_tensor(0.5, dtype=DEFAULT_FLOAT_DTYPE_TF)
    expected = 1.2661341104158121 # From BEAST 2 BirthDeathGerhard08ModelTest

    dist = BirthDeath(6, birth_diff_rate, relative_death_rate)
    res = dist.log_prob(tree)

    assert_allclose(res, expected, rtol=1e-6)