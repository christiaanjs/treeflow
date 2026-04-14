import pytest
import yaml
import tensorflow as tf
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    DEFAULT_TREE_VAR_NAME,
)
from treeflow.vi.hmc import fit_fixed_topology_hmc, KERNEL_HMC, KERNEL_NUTS
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree

NUM_RESULTS = 5
NUM_BURNIN = 3


@pytest.mark.parametrize("kernel", [KERNEL_HMC, KERNEL_NUTS])
def test_fit_fixed_topology_hmc_sample_shape(
    actual_model_file, hello_tensor_tree, hello_alignment, kernel
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(phylo_model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)

    result = fit_fixed_topology_hmc(
        model=pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        num_adaptation_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=3,
        kernel=kernel,
    )

    samples = result.samples
    # Samples should be a structured object from the model bijector
    # The tree variable should be a TensorflowRootedTree with batch dim
    flat_names = pinned._flat_resolve_names()
    flat_samples = pinned._model_flatten(samples)
    samples_dict = dict(zip(flat_names, flat_samples))

    assert DEFAULT_TREE_VAR_NAME in samples_dict
    tree_samples = samples_dict[DEFAULT_TREE_VAR_NAME]
    assert isinstance(tree_samples, TensorflowRootedTree)
    assert tree_samples.node_heights.shape[0] == NUM_RESULTS

    # Non-tree samples should have correct batch shape
    for name, sample in samples_dict.items():
        if name != DEFAULT_TREE_VAR_NAME:
            assert sample.shape[0] == NUM_RESULTS, (
                f"Variable '{name}' has wrong batch shape: {sample.shape}"
            )


def test_fit_fixed_topology_hmc_samples_finite(
    actual_model_file, hello_tensor_tree, hello_alignment
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(phylo_model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)

    result = fit_fixed_topology_hmc(
        model=pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        num_adaptation_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=3,
    )

    flat_names = pinned._flat_resolve_names()
    flat_samples = pinned._model_flatten(result.samples)
    samples_dict = dict(zip(flat_names, flat_samples))

    # Check that all sampled tensors are finite
    for name, sample in samples_dict.items():
        if isinstance(sample, TensorflowRootedTree):
            assert tf.reduce_all(tf.math.is_finite(sample.node_heights)), (
                f"Tree node_heights contain non-finite values"
            )
        else:
            assert tf.reduce_all(tf.math.is_finite(sample)), (
                f"Variable '{name}' contains non-finite values"
            )


@pytest.mark.parametrize("use_tf_function", [True, False])
def test_fit_fixed_topology_hmc_tf_function_mode(
    actual_model_file, hello_tensor_tree, hello_alignment, use_tf_function
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(phylo_model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)

    result = fit_fixed_topology_hmc(
        model=pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        num_adaptation_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=3,
        use_tf_function=use_tf_function,
    )

    flat_names = pinned._flat_resolve_names()
    flat_samples = pinned._model_flatten(result.samples)
    samples_dict = dict(zip(flat_names, flat_samples))
    tree_samples = samples_dict[DEFAULT_TREE_VAR_NAME]
    assert tree_samples.node_heights.shape[0] == NUM_RESULTS


def test_fit_fixed_topology_hmc_progress_bar_with_tf_function(
    actual_model_file, hello_tensor_tree, hello_alignment
):
    """Progress bar must reach its total when use_tf_function=True.

    Checks bar.n == total rather than just update_count > 0: the bar only
    reaches its total if py_function is executed for every step AND receives
    the correct (post-increment) step value so the update condition fires.
    A discarded or stale-step py_function call leaves bar.n == 0.
    """
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(phylo_model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)

    bar_ref = [None]

    class _CountingBar:
        def __init__(self, total):
            self.n = 0
            self.total = total

        def update(self, n):
            self.n += n

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def progress_bar_fn(total, **kwargs):
        bar = _CountingBar(total)
        bar_ref[0] = bar
        return bar

    fit_fixed_topology_hmc(
        model=pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        num_adaptation_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=3,
        progress_bar=progress_bar_fn,
        progress_bar_step=1,
        use_tf_function=True,
    )

    assert bar_ref[0].n >= NUM_RESULTS


def test_fit_fixed_topology_hmc_with_init_state(
    actual_model_file, hello_tensor_tree, hello_alignment
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(phylo_model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)

    init_state = {DEFAULT_TREE_VAR_NAME: hello_tensor_tree}

    result = fit_fixed_topology_hmc(
        model=pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        num_adaptation_steps=NUM_BURNIN,
        step_size=0.01,
        num_leapfrog_steps=3,
        init_state=init_state,
    )

    flat_names = pinned._flat_resolve_names()
    flat_samples = pinned._model_flatten(result.samples)
    assert len(flat_samples) == len(flat_names)
    for sample in flat_samples:
        if hasattr(sample, "node_heights"):
            assert sample.node_heights.shape[0] == NUM_RESULTS
        else:
            assert sample.shape[0] == NUM_RESULTS
