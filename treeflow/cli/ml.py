from functools import partial
import click
import yaml
import pickle
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.cli.inference_common import (
    optimizer_builders,
    ROBUST_ADAM_KEY,
    parse_init_values,
    EXAMPLE_PHYLO_MODEL_DICT,
    get_tree_vars,
    write_trees,
)
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    DEFAULT_TREE_VAR_NAME,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick, write_tensor_trees
from treeflow.evolution.seqio import Alignment
from treeflow.model.io import write_samples_to_file
from treeflow.model.ml import fit_fixed_topology_maximum_likelihood_sgd


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
        list(optimizer_builders.keys()),
    ),
    default=ROBUST_ADAM_KEY,
)
@click.option("-r", "--learning-rate", required=True, type=float, default=1e-3)
@click.option(
    "--init-values",
    required=False,
    type=str,
)
@click.option("--trace-output", required=False, type=click.Path())
@click.option("--variables-output", required=False, type=click.Path())
@click.option("--tree-output", required=False, type=click.Path())
def treeflow_ml(
    input,
    topology,
    num_steps,
    optimizer,
    model_file,
    learning_rate,
    init_values,
    trace_output,
    variables_output,
    tree_output,
):
    optimizer = optimizer_builders[optimizer](learning_rate=learning_rate)

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
        init = None
    else:
        init = {
            key: value for key, value in init_values_dict.items() if key in model_names
        }
        init[DEFAULT_TREE_VAR_NAME] = tree

    print(f"Running ML for up to {num_steps} iterations...")
    variables, trace, bijector = fit_fixed_topology_maximum_likelihood_sgd(
        model=pinned_model,
        topologies={DEFAULT_TREE_VAR_NAME: tree.topology},
        num_steps=num_steps,
        init=init,
    )
    print("Inference complete")
    trace_length = trace.log_likelihood.shape[0]
    if trace_length == num_steps:
        print("Optimization didn't converge")
    else:
        print(f"Optimization converged after {trace_length} steps")

    if trace_output is not None:
        print(f"Saving trace to {trace_output}...")
        with open(trace_output, "wb") as f:
            pickle.dump(trace, f)

    if variables_output is not None or tree_output is not None:
        variables_b = tf.nest.map_structure(partial(tf.expand_dims, axis=0), variables)
        variables_dict = variables_b._asdict()
        tree_vars = get_tree_vars(phylo_model)

        tree_samples = dict()
        for var in tree_vars:
            tree_samples[var] = variables_dict.pop(var)
        if variables_output is not None:
            print(f"Saving variables to {variables_output}...")
            write_samples_to_file(
                variables_b,
                pinned_model,
                variables_output,
                vars=variables_dict.keys(),
                tree_vars={DEFAULT_TREE_VAR_NAME: tree_samples[DEFAULT_TREE_VAR_NAME]},
            )

        if tree_output is not None:
            print(f"Saving tree samples to {tree_output}...")
            write_trees(tree_samples, topology, tree_output)
