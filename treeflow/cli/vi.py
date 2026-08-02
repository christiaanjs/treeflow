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
    PhyloModelParseError,
)
from treeflow.model.approximation import (
    get_fixed_topology_mean_field_approximation,
    get_fixed_topology_inverse_autoregressive_flow_approximation,
    get_fixed_topology_full_rank_approximation,
    get_fixed_topology_root_full_rank_approximation,
)
from treeflow.vi.fixed_topology_advi import fit_fixed_topology_variational_approximation
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick, TreeParseError
from treeflow.evolution.seqio import Alignment, AlignmentParseError
from treeflow.model.io import write_samples_to_file
from treeflow.vi.convergence_criteria import NonfiniteConvergenceCriterion
from treeflow.vi.util import VIResults
from treeflow.vi.plotting import plot_parameter_traces
from treeflow.cli.inference_common import (
    optimizer_builders,
    ROBUST_ADAM_KEY,
    parse_init_values,
    EXAMPLE_PHYLO_MODEL_DICT,
    get_tree_vars,
    write_trees,
    ALIGNMENT_FORMATS,
    DEFAULT_ALIGNMENT_FORMAT,
    InitialValueParseError,
)

convergence_criterion_classes = {"nonfinite": NonfiniteConvergenceCriterion}
approximation_builders = dict(
    full_rank=get_fixed_topology_full_rank_approximation,
    mean_field=get_fixed_topology_mean_field_approximation,
    iaf=get_fixed_topology_inverse_autoregressive_flow_approximation,
    root_full_rank=get_fixed_topology_root_full_rank_approximation,
)


@click.group()
def treeflow_vi():
    """
    Fixed-topology variational Bayesian inference for phylogenetic models.
    """


@treeflow_vi.command("run")
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
    default="full_rank",
    help="Variational approximation type",
    show_default=True,
)
@click.option(
    "--mean-field-vars",
    required=False,
    type=str,
    help="Comma-separated model variables to keep out of the full-covariance "
    "block (root_full_rank only), e.g. a per-branch relaxed clock rate whose "
    "dimension grows with the tree",
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
    show_default=True,
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
    "--resume-from-trace",
    required=False,
    type=click.Path(exists=True),
    help=(
        "Path to a pickled optimization trace (as saved by --trace-output) to resume "
        "from. The variational parameters are warm-started from the last step of the "
        "trace; this must come from a run with the same --variational-approximation, "
        "model and topology."
    ),
)
@click.option(
    "--max-trace-coords",
    required=False,
    type=click.IntRange(min=1),
    help=(
        "If set, --trace-output records only up to this many coordinates per "
        "variational parameter at each step (its last coordinate -- the root, for "
        "a node-height vector -- plus a random selection of the rest), instead of "
        "the full tensor. Keeps trace memory bounded regardless of parameter size "
        "(e.g. a full-rank scale matrix, which is quadratic in the number of free "
        "model dimensions), at the cost of a coarser trace for diagnostics. "
        "`treeflow_vi plot` labels such traces by parameter coordinate."
    ),
)
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
    help="Path to save tree samples to in Nexus format",
)
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
def run(
    input,
    topology,
    num_steps,
    optimizer,
    model_file,
    variational_approximation,
    mean_field_vars,
    learning_rate,
    init_values,
    seed,
    trace_output,
    resume_from_trace,
    max_trace_coords,
    samples_output,
    tree_samples_output,
    n_output_samples,
    convergence_criterion,
    elbo_samples,
    progress_bar,
    subnewick_format,
    alignment_format,
):
    """
    Perform fixed-topology variational Bayesian inference for a phylogenetic model
    with a given tree topology and multiple sequence alignment.

    The tree prior and substitution model used can be specified using the TreeFlow
    YAML model definition format (see the package documentation).
    """
    optimizer = optimizer_builders[optimizer](learning_rate=learning_rate)

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

    print(f"Parsing initial values...")
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

    if mean_field_vars:
        if variational_approximation != "root_full_rank":
            raise click.ClickException(
                "--mean-field-vars is only meaningful for the root_full_rank "
                f"approximation, not {variational_approximation}"
            )
        names = [name.strip() for name in mean_field_vars.split(",") if name.strip()]
        unknown = sorted(set(names) - model_names)
        if unknown:
            raise click.ClickException(
                f"--mean-field-vars {unknown} are not model variables; model "
                f"variables are {sorted(model_names)}"
            )
        approx_kwargs["mean_field_vars"] = names

    if resume_from_trace is None:
        resume_from_variables = None
    else:
        print(f"Resuming from trace {resume_from_trace}...")
        with open(resume_from_trace, "rb") as f:
            previous_trace: VIResults = pickle.load(f)
        resume_from_variables = {
            name: tf.convert_to_tensor(value)[-1]
            for name, value in previous_trace.parameters.items()
        }

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
        resume_from_variables=resume_from_variables,
        max_trace_coords=max_trace_coords,
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


@treeflow_vi.command("plot")
@click.option(
    "-t",
    "--trace",
    required=True,
    type=click.Path(exists=True),
    help="Pickled optimization trace, as saved by `treeflow_vi run --trace-output`",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(),
    help="Path to save the plot to (format inferred from extension, e.g. .png/.pdf)",
)
@click.option(
    "--sample/--full",
    default=False,
    show_default=True,
    help=(
        "Plot a small representative sample of coordinates in a single axis "
        "instead of one subplot per variable"
    ),
)
@click.option(
    "--coords-per-var",
    type=int,
    default=1,
    show_default=True,
    help="Coordinates sampled per non-tree vector variable (--sample only)",
)
@click.option(
    "--tree-vars",
    type=str,
    default=DEFAULT_TREE_VAR_NAME,
    show_default=True,
    help="Comma-separated name substrings identifying node-height vector variables",
)
@click.option(
    "--tree-coords",
    type=int,
    default=3,
    show_default=True,
    help=(
        "Coordinates sampled per tree variable: the root plus this many minus one "
        "internal nodes (--sample only)"
    ),
)
@click.option(
    "--max-individual-lines",
    type=int,
    default=16,
    show_default=True,
    help=(
        "Coordinate count above which a variable is summarised as a mean/min-max "
        "envelope instead of one line per coordinate (--full only)"
    ),
)
@click.option(
    "--ncols",
    type=int,
    default=3,
    show_default=True,
    help="Number of subplot columns (--full only)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Plot title (--sample only)",
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    show_default=True,
    help="Resolution of the saved figure",
)
def plot(
    trace,
    output,
    sample,
    coords_per_var,
    tree_vars,
    tree_coords,
    max_individual_lines,
    ncols,
    title,
    dpi,
):
    """
    Plot the optimization trace of variational parameters saved by
    `treeflow_vi run --trace-output`.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")

    print(f"Loading trace {trace}...")
    with open(trace, "rb") as f:
        vi_results: VIResults = pickle.load(f)

    axes = plot_parameter_traces(
        vi_results.parameters,
        sample=sample,
        coords_per_var=coords_per_var,
        tree_vars=tuple(part.strip() for part in tree_vars.split(",") if part.strip()),
        tree_coords=tree_coords,
        title=title,
        max_individual_lines=max_individual_lines,
        ncols=ncols,
        # `None` unless the trace was written with `run --max-trace-coords`, in
        # which case it maps traced positions back to variable coordinates.
        parameter_coords=vi_results.parameter_coords,
    )
    figure = np.atleast_1d(axes)[0].figure

    print(f"Saving plot to {output}...")
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    print("Exiting...")
