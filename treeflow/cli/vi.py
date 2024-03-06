from __future__ import annotations

import click
import pickle
import yaml
import typing as tp
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    DEFAULT_TREE_VAR_NAME,
)
from treeflow.model.approximation import (
    get_fixed_topology_mean_field_approximation,
    get_fixed_topology_inverse_autoregressive_flow_approximation,
    get_inverse_autoregressive_flow_approximation,
)
from treeflow.vi.fixed_topology_advi import fit_fixed_topology_variational_approximation
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment
from treeflow.model.io import write_samples_to_file
from treeflow.vi.convergence_criteria import NonfiniteConvergenceCriterion
from treeflow.vi.util import VIResults
from treeflow.cli.inference_common import (
    optimizer_builders,
    ROBUST_ADAM_KEY,
    parse_init_values,
    EXAMPLE_PHYLO_MODEL_DICT,
    get_tree_vars,
    write_trees,
)

convergence_criterion_classes = {"nonfinite": NonfiniteConvergenceCriterion}
approximation_builders = dict(
    mean_field=get_fixed_topology_mean_field_approximation,
    iaf=get_fixed_topology_inverse_autoregressive_flow_approximation,
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
    "--variational-approximation",
    "-va",
    type=click.Choice(list(approximation_builders.keys())),
    required=True,
    default="mean_field",
    help="Variational approximation type",
    show_default=True,
)
@click.option(
    "-n",
    "--num-steps",
    required=True,
    type=int,
    help="Number of VI iterations",
    default=40000,
    show_default=True,
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
@click.option(
    "--init-values",
    required=False,
    type=str,
    help="Initial values in the format 'scalar_parameter=value1,vector_parameter=value2a|value2b'",
)
@click.option("-s", "--seed", required=False, type=int)
@click.option(
    "--trace-output",
    required=False,
    type=click.Path(),
    help="Path to save pickled optimization trace",
)
@click.option(
    "--samples-output",
    required=False,
    type=click.Path(),
    help="Path to save parameter samples in CSV format",
)
@click.option("--tree-samples-output", required=False, type=click.Path())
@click.option(
    "--n-output-samples",
    required=True,
    type=int,
    default=200,
    help="Number of samples to use for outputs",
    show_default=True,
)
@click.option(
    "-r", "--learning-rate", required=True, type=float, default=1e-3, show_default=True
)
@click.option(
    "-c",
    "--convergence-criterion",
    required=False,
    type=click.Choice(list(convergence_criterion_classes.keys())),
)
@click.option(
    "--elbo-samples",
    required=True,
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Number of samples to use in displayed estimate of evidence lower bound",
)
@click.option("--progress-bar/--no-progress-bar", default=True)
@click.option(
    "--subnewick-format",
    type=int,
    required=True,
    default=0,
    help="Subnewick format (see `ete3.Tree`)",
    show_default=True,
)
def treeflow_vi(
    input,
    topology,
    num_steps,
    optimizer,
    model_file,
    variational_approximation,
    learning_rate,
    init_values,
    seed,
    trace_output,
    samples_output,
    tree_samples_output,
    n_output_samples,
    convergence_criterion,
    elbo_samples,
    progress_bar,
    subnewick_format,
):
    optimizer = optimizer_builders[optimizer](learning_rate=learning_rate)

    print(f"Parsing topology {topology}")
    tree = convert_tree_to_tensor(
        parse_newick(topology, subnewick_format=subnewick_format)
    )

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
    phylo_model = PhyloModel(model_dict)
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

    if convergence_criterion is not None:
        convergence_criterion_instance = convergence_criterion_classes[
            convergence_criterion
        ]()
    else:
        convergence_criterion_instance = None

    if variational_approximation == "iaf":
        approx_kwargs = dict(hidden_units_per_layer=tree.taxon_count)
    else:
        approx_kwargs = dict()

    print(f"Running VI for {num_steps} iterations...")
    vi_res: tp.Tuple[object, VIResults] = fit_fixed_topology_variational_approximation(
        model=pinned_model,
        topologies={DEFAULT_TREE_VAR_NAME: tree.topology},
        init_loc=init_loc,
        optimizer=optimizer,
        num_steps=num_steps,
        convergence_criterion=convergence_criterion_instance,
        seed=seed,
        progress_bar=progress_bar,
        approx_fn=approximation_builders[variational_approximation],
        approx_kwargs=approx_kwargs,
    )
    approx, trace = vi_res
    print("Inference complete")

    inference_steps = trace.loss.shape[0]
    print(f"Ran inference for {inference_steps} iterations")
    elbo_estimate = -tf.reduce_sum(trace.loss[-elbo_samples:]).numpy()
    print(f"ELBO estimate: {elbo_estimate}")

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
                samples,
                pinned_model,
                samples_output,
                vars=samples_dict.keys(),
                tree_vars={DEFAULT_TREE_VAR_NAME: tree_samples[DEFAULT_TREE_VAR_NAME]},
            )

        if tree_samples_output is not None:
            print(f"Saving tree samples to {tree_samples_output}...")
            write_trees(tree_samples, topology, tree_samples_output)

    print("Exiting...")
