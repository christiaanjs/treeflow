from __future__ import annotations

import click
import pickle
import yaml
import typing as tp
import tensorflow as tf
import tensorflow.keras.optimizers as keras_optimizers
from treeflow import DEFAULT_FLOAT_DTYPE_TF
import treeflow.evolution.substitution as substitution
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    DEFAULT_TREE_VAR_NAME,
)
from treeflow.vi.fixed_topology_advi import fit_fixed_topology_variational_approximation
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick, write_tensor_trees
from treeflow.evolution.seqio import Alignment
from treeflow.model.io import write_samples_to_file


_ADAM_KEY = "adam"
optimizer_classes = {_ADAM_KEY: keras_optimizers.Adam}

EXAMPLE_PHYLO_MODEL_DICT = dict(
    tree=dict(coalescent=dict(pop_size=dict(exponential=dict(rate=0.1)))),
    clock=dict(strict=dict(rate=1e-3)),
    substitution="jc",
)


def parse_init_values(init_values_string: str) -> tp.Dict[str, tf.Tensor]:
    str_dict = dict(item.split("=") for item in init_values_string.split(","))
    return {key: float(value) for key, value in str_dict.items()}


def get_tree_vars(model: PhyloModel) -> set[str]:
    tree_vars = {DEFAULT_TREE_VAR_NAME}
    if model.relaxed_clock():
        tree_vars.add("branch_rates")
    return tree_vars


def write_trees(
    tree_var_samples: dict[str, tf.Tensor], topology_file, output_file
) -> None:
    branch_lengths = tree_var_samples.pop(DEFAULT_TREE_VAR_NAME).branch_lengths
    write_tensor_trees(
        topology_file, branch_lengths, output_file, branch_metadata=tree_var_samples
    )


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Alignment file (FASTA format)",
)
@click.option(
    "-t",
    "--topology",
    required=True,
    type=click.Path(exists=True),
    help="Topology file",
)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(exists=True),
    help="YAML model definition file",
)
@click.option(
    "-n", "--num-steps", required=True, type=int, help="Number of VI iterations"
)
@click.option(
    "-o",
    "--optimizer",
    required=True,
    type=click.Choice(
        list(optimizer_classes.keys()),
    ),
    default=_ADAM_KEY,
)
@click.option(
    "--init-values",
    required=False,
    type=str,
)
@click.option("--trace-output", required=False, type=click.Path())
@click.option("--samples-output", required=False, type=click.Path())
@click.option("--tree-samples-output", required=False, type=click.Path())
@click.option("--n-output-samples", required=False, type=int, default=200)
@click.option("-r", "--learning-rate", required=True, type=float, default=1e-3)
def treeflow_vi(
    input,
    topology,
    num_steps,
    optimizer,
    model_file,
    learning_rate,
    init_values,
    trace_output,
    samples_output,
    tree_samples_output,
    n_output_samples,
):
    optimizer = optimizer_classes[optimizer](learning_rate=learning_rate)

    print(f"Parsing topology {topology}")
    tree = convert_tree_to_tensor(parse_newick(topology))

    print(f"Parsing alignment {input}")
    alignment = Alignment(input).get_compressed_alignment()
    encoded_sequences = alignment.get_encoded_sequence_tensor(tree.taxon_set)
    pattern_counts = alignment.get_weights_tensor()

    print(f"Parsing initial values...")
    init_values_dict = (
        None
        if init_values is None
        else {
            key: tf.constant(value, dtype=DEFAULT_FLOAT_DTYPE_TF)
            for key, value in parse_init_values(init_values).items()
        }
    )

    if model_file is None:
        model_dict = EXAMPLE_PHYLO_MODEL_DICT
    else:
        with open(model_file) as f:
            model_dict = yaml.safe_load(f)
    phylo_model = model = PhyloModel(model_dict)
    model = phylo_model_to_joint_distribution(
        phylo_model, tree, alignment, pattern_counts=pattern_counts
    )
    pinned_model = model.experimental_pin(alignment=encoded_sequences)
    model_names = set(pinned_model._flat_resolve_names())

    if init_values_dict is None:
        init_loc = None
    else:
        init_loc = {
            key: value for key, value in init_values_dict.items() if key in model_names
        }
        init_loc[DEFAULT_TREE_VAR_NAME] = tree

    print(f"Running VI for {num_steps} iterations...")
    approx, trace = fit_fixed_topology_variational_approximation(
        model=pinned_model,
        topologies={DEFAULT_TREE_VAR_NAME: tree.topology},
        init_loc=init_loc,
        optimizer=optimizer,
        num_steps=num_steps,
    )
    print("Inference complete")

    if trace_output is not None:
        print(f"Saving trace to {trace_output}...")
        with open(trace_output, "wb") as f:
            pickle.dump(trace, f)

    if samples_output is not None or tree_samples_output is not None:
        print("Sampling fitted approximation...")
        samples = approx.sample(n_output_samples)
        samples_dict = samples._asdict()
        tree_vars = get_tree_vars(phylo_model)

        tree_samples = dict()
        for var in tree_vars:
            tree_samples[var] = samples_dict.pop(var)

        if samples_output is not None:
            print(f"Saving samples to {samples_output}...")
            write_samples_to_file(
                samples, pinned_model, samples_output, vars=samples_dict.keys()
            )

        if tree_samples_output is not None:
            print(f"Saving tree samples to {tree_samples_output}...")
            write_trees(tree_samples, topology, tree_samples_output)

    print("Exiting...")
