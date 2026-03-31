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
