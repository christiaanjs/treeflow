import pytest
import numpy as np
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree import TensorflowRootedTree
from treeflow.evolution import Alignment, get_transition_probabilities_tree, JC
from treeflow.distributions import ConstantCoalescent, LeafCTMC
from treeflow.model.approximation import get_fixed_topology_mean_field_approximation
from treeflow.vi import estimate_log_ml_importance_sampling
from tensorflow_probability.python.distributions import (
    JointDistributionNamed,
    Exponential,
    Sample,
)


@pytest.mark.parametrize("vectorize_log_prob", [False, True])
def test_marginal_likelihood(
    hello_alignment: Alignment,
    hello_tensor_tree: TensorflowRootedTree,
    vectorize_log_prob,
):

    clock_rate = tf.constant(1e-2, dtype=DEFAULT_FLOAT_DTYPE_TF)
    subst_model = JC()

    def get_sequence_distribution(tree: TensorflowRootedTree):
        transition_probs = get_transition_probabilities_tree(
            tree.get_unrooted_tree() * clock_rate, subst_model
        )
        return Sample(
            LeafCTMC(transition_probs, JC.frequencies()), hello_alignment.site_count
        )

    model = JointDistributionNamed(
        dict(
            pop_size=Exponential(tf.constant(1.0, dtype=DEFAULT_FLOAT_DTYPE_TF)),
            tree=lambda pop_size: ConstantCoalescent(
                hello_tensor_tree.taxon_count,
                pop_size,
                hello_tensor_tree.sampling_times,
                name="tree",
            ),
            sequences=get_sequence_distribution,
        ),
    )
    model_pinned = model.experimental_pin(
        sequences=hello_alignment.get_encoded_sequence_tensor(
            hello_tensor_tree.taxon_set
        )
    )
    approx, variable_dict = get_fixed_topology_mean_field_approximation(
        model_pinned, dict(tree=hello_tensor_tree.topology)
    )
    res = estimate_log_ml_importance_sampling(
        model_pinned, approx, n_samples=10, vectorize_log_prob=vectorize_log_prob
    )
    assert tuple(res.shape) == ()
    assert np.isfinite(res.numpy())
