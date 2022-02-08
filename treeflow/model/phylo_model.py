import typing as tp
from functools import partial
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.evolution.substitution import JC
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.distributions.sample_weighted import SampleWeighted

DEFAULT_TREE_NAME = "tree"


def alignment_func(
    tree: TensorflowRootedTree,
    rates: tf.Tensor,
    site_count: int,
    weights: tp.Optional[tf.Tensor] = None,
):
    unrooted_time_tree = tree.get_unrooted_tree()
    distance_tree = unrooted_time_tree.with_branch_lengths(
        unrooted_time_tree.branch_lengths * rates
    )
    subst_model = JC()
    frequencies = subst_model.frequencies(dtype=rates.dtype)
    transition_probs = get_transition_probabilities_tree(
        distance_tree,
        subst_model,
        frequencies=frequencies,
    )
    leaf_ctmc = LeafCTMC(transition_probs, frequencies=frequencies)
    sample_shape = (site_count,)
    if weights is None:
        return tfd.Sample(leaf_ctmc, sample_shape=sample_shape)
    else:
        return SampleWeighted(leaf_ctmc, weights=weights, sample_shape=sample_shape)


def strict_clock_fixed_alignment_func(rate, **kwargs):
    def _alignment_func(tree):
        return alignment_func(tree, rate, **kwargs)

    return _alignment_func


def get_example_phylo_model(
    taxon_count: int,
    site_count: int,
    sampling_times: tf.Tensor,
    init_values: tp.Optional[tp.Dict[str, float]] = None,
    dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    tree_name: str = DEFAULT_TREE_NAME,
    pattern_counts: tp.Optional[tf.Tensor] = None,
) -> tfd.JointDistribution:
    if init_values is None:
        init_values = {}

    constant = partial(tf.constant, dtype=dtype)
    rate = constant(init_values.get("rate", 1e-3))

    model_dict = dict(
        pop_size=tfd.Exponential(rate=constant(0.1), name="pop_size"),
        tree=lambda pop_size: ConstantCoalescent(
            taxon_count=taxon_count,
            pop_size=pop_size,
            sampling_times=sampling_times,
            tree_name=tree_name,
        ),
        alignment=strict_clock_fixed_alignment_func(
            rate, site_count=site_count, weights=pattern_counts
        ),
    )
    return tfd.JointDistributionNamed(model_dict)
