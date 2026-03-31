from __future__ import annotations

import click
import yaml
import typing as tp
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    DEFAULT_TREE_VAR_NAME,
    PhyloModelParseError,
)
from treeflow.vi.hmc import (
    fit_fixed_topology_hmc,
    KERNEL_HMC,
    KERNEL_NUTS,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick, TreeParseError
from treeflow.evolution.seqio import Alignment, AlignmentParseError
from treeflow.model.io import write_samples_to_file
from treeflow.cli.inference_common import (
    parse_init_values,
    EXAMPLE_PHYLO_MODEL_DICT,
    get_tree_vars,
    write_trees,
    ALIGNMENT_FORMATS,
    DEFAULT_ALIGNMENT_FORMAT,
    InitialValueParseError,
)

KERNEL_CHOICES = [KERNEL_HMC, KERNEL_NUTS]


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Alignment file",
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
    "-n",
    "--num-results",
    required=True,
    type=int,
    default=1000,
    show_default=True,
    help="Number of MCMC samples to collect after burn-in",
)
@click.option(
    "--num-burnin-steps",
    required=True,
    type=int,
    default=500,
    show_default=True,
    help="Number of burn-in steps (discarded)",
)
@click.option(
    "--num-adaptation-steps",
    required=False,
    type=int,
    default=None,
    help="Steps for dual-averaging step size adaptation (default: num-burnin-steps)",
)
@click.option(
    "--step-size",
    required=True,
    type=float,
    default=0.01,
    show_default=True,
    help="Initial leapfrog step size",
)
@click.option(
    "--num-leapfrog-steps",
    required=True,
    type=int,
    default=10,
    show_default=True,
    help="Number of leapfrog steps per HMC proposal (ignored for NUTS)",
)
@click.option(
    "--kernel",
    required=True,
    type=click.Choice(KERNEL_CHOICES),
    default=KERNEL_HMC,
    show_default=True,
    help="MCMC kernel type",
)
@click.option(
    "--init-values",
    required=False,
    type=str,
    help="Initial values in the format 'scalar_parameter=value1,vector_parameter=value2a|value2b'",
)
@click.option("-s", "--seed", required=False, type=int)
@click.option(
    "--samples-output",
    required=False,
    type=click.Path(),
    help="Path to save parameter samples in CSV format",
)
@click.option(
    "--tree-samples-output",
    required=False,
    type=click.Path(),
    help="Path to save tree samples in Nexus format",
)
@click.option(
    "--alignment-format",
    required=True,
    type=click.Choice(list(ALIGNMENT_FORMATS.keys())),
    default=DEFAULT_ALIGNMENT_FORMAT,
    show_default=True,
    help="File format for alignment",
)
@click.option(
    "--subnewick-format",
    type=int,
    required=True,
    default=0,
    help="Subnewick format (see `ete3.Tree`)",
    show_default=True,
)
def treeflow_hmc(
    input,
    topology,
    model_file,
    num_results,
    num_burnin_steps,
    num_adaptation_steps,
    step_size,
    num_leapfrog_steps,
    kernel,
    init_values,
    seed,
    samples_output,
    tree_samples_output,
    subnewick_format,
    alignment_format,
):
    """
    Perform fixed-topology Hamiltonian Monte Carlo Bayesian phylogenetic inference
    with a given tree topology and multiple sequence alignment.

    The tree prior and substitution model used can be specified using the TreeFlow
    YAML model definition format (see the package documentation).
    """
    print(f"Parsing topology {topology}")
    try:
        tree = convert_tree_to_tensor(
            parse_newick(topology, subnewick_format=subnewick_format)
        )
    except TreeParseError as ex:
        raise click.ClickException(str(ex))

    print(f"Parsing alignment {input}")
    try:
        alignment = Alignment(
            input, format=ALIGNMENT_FORMATS[alignment_format]
        ).get_compressed_alignment()
    except AlignmentParseError as ex:
        raise click.ClickException(str(ex))
    encoded_sequences = alignment.get_encoded_sequence_tensor(tree.taxon_set)
    pattern_counts = alignment.get_weights_tensor()

    print("Parsing model...")
    if model_file is None:
        model_dict = EXAMPLE_PHYLO_MODEL_DICT
    else:
        with open(model_file) as f:
            model_dict = yaml.safe_load(f)
    try:
        phylo_model = PhyloModel(model_dict)
    except PhyloModelParseError as ex:
        raise click.ClickException(str(ex))
    model = phylo_model_to_joint_distribution(
        phylo_model, tree, alignment, pattern_counts=pattern_counts
    )
    pinned_model = model.experimental_pin(alignment=encoded_sequences)
    model_names = set(pinned_model._flat_resolve_names())

    print("Parsing initial values...")
    try:
        init_values_dict = (
            None
            if init_values is None
            else {
                key: tf.constant(value, dtype=DEFAULT_FLOAT_DTYPE_TF)
                for key, value in parse_init_values(
                    init_values, model_names=model_names
                ).items()
            }
        )
    except InitialValueParseError as ex:
        raise click.ClickException(str(ex))

    if init_values_dict is None:
        init_state = None
    else:
        init_state = {
            key: value
            for key, value in init_values_dict.items()
            if key in model_names
        }
        init_state[DEFAULT_TREE_VAR_NAME] = tree

    total_steps = num_results + num_burnin_steps
    print(
        f"Running HMC: {num_burnin_steps} burn-in + {num_results} sampling steps "
        f"({total_steps} total)..."
    )
    hmc_res = fit_fixed_topology_hmc(
        model=pinned_model,
        topologies={DEFAULT_TREE_VAR_NAME: tree.topology},
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_adaptation_steps=num_adaptation_steps,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        kernel=kernel,
        init_state=init_state,
        seed=seed,
    )
    print("Sampling complete")

    if samples_output is not None or tree_samples_output is not None:
        constrained_samples = hmc_res.samples
        names = pinned_model._flat_resolve_names()
        flat_samples = pinned_model._model_flatten(constrained_samples)
        samples_dict = dict(zip(names, flat_samples))

        tree_vars = get_tree_vars(phylo_model)
        tree_samples = {var: samples_dict.pop(var) for var in tree_vars if var in samples_dict}

        if samples_output is not None:
            print(f"Saving samples to {samples_output}...")
            write_samples_to_file(
                constrained_samples,
                pinned_model,
                samples_output,
                vars=samples_dict.keys(),
                tree_vars={DEFAULT_TREE_VAR_NAME: tree_samples[DEFAULT_TREE_VAR_NAME]}
                if DEFAULT_TREE_VAR_NAME in tree_samples
                else None,
            )

        if tree_samples_output is not None:
            print(f"Saving tree samples to {tree_samples_output}...")
            write_trees(tree_samples, topology, tree_samples_output)

    print("Exiting...")
