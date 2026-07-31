import pytest
import typing as tp
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick
from treeflow.model.approximation import (
    get_mean_field_approximation,
    get_fixed_topology_mean_field_approximation,
    get_fixed_topology_inverse_autoregressive_flow_approximation,
    get_fixed_topology_root_full_rank_approximation,
)
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.distributions.tree.birthdeath.yule import Yule
from treeflow_test_helpers.tree_helpers import TreeTestData, data_to_tensor_tree
from treeflow.model.phylo_model import (
    PhyloModel,
    get_sequence_distribution,
    phylo_model_to_joint_distribution,
)

_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def test_get_mean_field_approximation():
    sample_size = 3
    model = tfd.JointDistributionNamed(
        dict(
            a=tfd.Normal(_constant(0.0), _constant(1.0)),
            b=lambda a: tfd.Dirichlet(tf.fill((sample_size,), _constant(2.0))),
            obs=lambda b: tfd.Independent(
                tfd.Normal(b, _constant(1.0)), reinterpreted_batch_ndims=1
            ),
        )
    )
    obs = _constant([-1.1, 2.1, 0.1])
    pinned = model.experimental_pin(obs=obs)
    approximation, variable_dict = get_mean_field_approximation(
        pinned, init_loc=dict(a=_constant(0.1)), dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    sample = approximation.sample()
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


@pytest.mark.parametrize("with_init_loc", [True, False])
def test_get_mean_field_approximation_tree(
    flat_tree_test_data: TreeTestData, with_init_loc: bool
):
    test_tree = data_to_tensor_tree(flat_tree_test_data)
    taxon_count = test_tree.taxon_count
    tree_name = "tree_dist_name"

    init_loc: tp.Optional[tp.Dict[str, object]]
    if with_init_loc:
        init_loc = dict(tree=test_tree)
    else:
        init_loc = None

    model = tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, test_tree.sampling_times, tree_name=tree_name
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0), tf.reduce_sum(tree.branch_lengths)
            ),
        )
    )
    obs = _constant([10.0])
    pinned = model.experimental_pin(obs=obs)
    approximation, variable_dict = get_fixed_topology_mean_field_approximation(
        pinned,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: test_tree.topology},
        init_loc=init_loc,
    )

    sample = approximation.sample()
    assert (
        tf.reduce_all(
            sample["tree"].topology.parent_indices == test_tree.topology.parent_indices
        )
        .numpy()
        .item()
    )
    assert_allclose(
        sample["tree"].sampling_times.numpy(), test_tree.sampling_times.numpy()
    )
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


@pytest.mark.parametrize("with_init_loc", [True, False])
def test_get_mean_field_approximation_tree_yule(
    tensor_constant, hello_newick_file, with_init_loc
):
    tree = convert_tree_to_tensor(parse_newick(hello_newick_file))
    tree_name = "tree_dist_name"
    model = tfd.JointDistributionNamed(
        {
            "rates": tfd.Sample(
                tfd.LogNormal(_constant(0.0), _constant(1.0)), tree.branch_lengths.shape
            ),
            "birth_rate": tfd.LogNormal(_constant(1.0), _constant(1.5)),
            tree_name: lambda birth_rate: Yule(
                tree.taxon_count, birth_rate, name=tree_name
            ),
            "a": lambda tree_dist_name, rates: tfd.Normal(
                tf.reduce_sum(tree_dist_name.branch_lengths * rates, axis=-1),
                _constant(1.0),
            ),
        }
    )

    if with_init_loc:
        init_loc = dict(tree=tree, birth_rate=_constant(2.0))
    else:
        init_loc = None

    approximation, variable_dict = get_fixed_topology_mean_field_approximation(
        model,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: tree.topology},
        init_loc=init_loc,
    )
    sample = approximation.sample()
    model_log_prob = model.log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


def test_get_iaf_approximation_tree(flat_tree_test_data: TreeTestData):
    test_tree = data_to_tensor_tree(flat_tree_test_data)
    taxon_count = test_tree.taxon_count.item()
    tree_name = "tree_dist_name"

    model = tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, test_tree.sampling_times, tree_name=tree_name
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0), tf.reduce_sum(tree.branch_lengths)
            ),
        )
    )
    obs = _constant([10.0])
    n_hidden_layers = 2
    n_iaf_bijectors = 2
    pinned = model.experimental_pin(obs=obs)
    (
        approximation,
        variable_dict,
    ) = get_fixed_topology_inverse_autoregressive_flow_approximation(
        pinned,
        taxon_count,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: test_tree.topology},
    )
    # network weights: (n_hidden_layers + 1 output) * (kernel + bias) * n_bijectors
    # + 2 affine-base variables (loc_var, log_scale_var)
    assert len(variable_dict) == n_hidden_layers * n_iaf_bijectors * 3 + 2
    sample = approximation.sample()
    assert (
        tf.reduce_all(
            sample["tree"].topology.parent_indices == test_tree.topology.parent_indices
        )
        .numpy()
        .item()
    )
    assert_allclose(
        sample["tree"].sampling_times.numpy(), test_tree.sampling_times.numpy()
    )
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


@pytest.mark.parametrize("with_init_loc", [True, False])
def test_get_root_full_rank_approximation_tree(
    flat_tree_test_data: TreeTestData, with_init_loc: bool
):
    test_tree = data_to_tensor_tree(flat_tree_test_data)
    taxon_count = test_tree.taxon_count
    tree_name = "tree_dist_name"

    init_loc: tp.Optional[tp.Dict[str, object]]
    if with_init_loc:
        init_loc = dict(tree=test_tree)
    else:
        init_loc = None

    model = tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, test_tree.sampling_times, tree_name=tree_name
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0), tf.reduce_sum(tree.branch_lengths)
            ),
        )
    )
    obs = _constant([10.0])
    pinned = model.experimental_pin(obs=obs)
    approximation, variable_dict = get_fixed_topology_root_full_rank_approximation(
        pinned,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: test_tree.topology},
        init_loc=init_loc,
    )

    sample = approximation.sample()
    assert (
        tf.reduce_all(
            sample["tree"].topology.parent_indices == test_tree.topology.parent_indices
        )
        .numpy()
        .item()
    )
    assert_allclose(
        sample["tree"].sampling_times.numpy(), test_tree.sampling_times.numpy()
    )
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


def test_root_full_rank_approximation_block_independence(
    flat_tree_test_data: TreeTestData,
):
    """The full-rank block (root height + non-tree variables) and the
    mean-field block (the tree's other node-height ratios) must be
    independent of each other: perturbing one must never move the other,
    and the full-rank block should show genuine cross-coordinate
    sensitivity (unlike a pure mean-field approximation)."""
    test_tree = data_to_tensor_tree(flat_tree_test_data)
    taxon_count = test_tree.taxon_count
    tree_name = "tree_dist_name"

    model = tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            clock_rate=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, test_tree.sampling_times, tree_name=tree_name
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0), tf.reduce_sum(tree.branch_lengths)
            ),
        )
    )
    obs = _constant([10.0])
    pinned = model.experimental_pin(obs=obs)
    approximation, variable_dict = get_fixed_topology_root_full_rank_approximation(
        pinned,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: test_tree.topology},
    )

    # Force real off-diagonal correlation into the full-rank block so any
    # leakage into the mean-field block would be detectable.
    raw_scale_var = variable_dict["root_full_rank_scale_raw:0"]
    raw_scale_var.assign(raw_scale_var + 2.0)

    chain = approximation.bijector
    permute_bijector = next(b for b in chain.bijectors if type(b).__name__ == "Permute")
    blockwise_bijector = next(
        b for b in chain.bijectors if type(b).__name__ == "_Blockwise"
    )

    full_rank_size = variable_dict["root_full_rank_loc:0"].shape[0]
    n_ratios = variable_dict["root_full_rank_ratio_loc:0"].shape[0]
    total_dim = full_rank_size + n_ratios

    def composed(x):
        return permute_bijector.forward(blockwise_bijector.forward(x))

    base = tf.zeros([total_dim], dtype=DEFAULT_FLOAT_DTYPE_TF)
    base_out = composed(base).numpy()
    eps = 1e-3
    jacobian = np.zeros((total_dim, total_dim))
    for i in range(total_dim):
        perturbed = tf.tensor_scatter_nd_update(base, [[i]], [_constant(eps)])
        jacobian[:, i] = (composed(perturbed).numpy() - base_out) / eps

    permutation = permute_bijector.permutation.numpy()
    ratio_out_idx = [i for i, p in enumerate(permutation) if p >= full_rank_size]
    fr_out_idx = [i for i, p in enumerate(permutation) if p < full_rank_size]

    leak_fr_into_ratio = jacobian[np.ix_(ratio_out_idx, range(full_rank_size))]
    leak_ratio_into_fr = jacobian[
        np.ix_(fr_out_idx, range(full_rank_size, total_dim))
    ]
    fr_into_fr = jacobian[np.ix_(fr_out_idx, range(full_rank_size))]
    ratio_into_ratio = jacobian[
        np.ix_(ratio_out_idx, range(full_rank_size, total_dim))
    ]

    assert_allclose(leak_fr_into_ratio, 0.0, atol=1e-8)
    assert_allclose(leak_ratio_into_fr, 0.0, atol=1e-8)
    assert np.abs(fr_into_fr).max() > 0.1  # genuine cross-coordinate correlation
    off_diagonal = ratio_into_ratio - np.diag(np.diag(ratio_into_ratio))
    assert_allclose(off_diagonal, 0.0, atol=1e-8)  # ratios stay mean-field
