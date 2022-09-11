import yaml
import tensorflow as tf
from tensorflow_probability.python.vi import fit_surrogate_posterior
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
)
from treeflow.model.approximation import get_fixed_topology_mean_field_approximation


def test_fit_surrogate_posterior_n_samples(
    actual_model_file, hello_tensor_tree, hello_alignment
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)

    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    pinned = dist.experimental_pin(alignment=encoded_sequences)
    approximation, variables_dict = get_fixed_topology_mean_field_approximation(
        pinned, topology_pins=dict(tree=hello_tensor_tree.topology)
    )
    optimizer = tf.optimizers.Adam(learning_rate=1e-2)
    num_steps = 11
    trace = fit_surrogate_posterior(
        pinned.unnormalized_log_prob,
        approximation,
        optimizer,
        num_steps,
        sample_size=3,
        importance_sample_size=4,
    )
    assert tuple(trace.shape) == (num_steps,)
