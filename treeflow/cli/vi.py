import click
import pickle
import typing as tp
import tensorflow as tf
import tensorflow.keras.optimizers as keras_optimizers
from treeflow import DEFAULT_FLOAT_DTYPE_TF
import treeflow.evolution.substitution as substitution
from treeflow.model.phylo_model import (
    get_example_phylo_model,
    DEFAULT_TREE_NAME,
)
from treeflow.vi.fixed_topology_advi import fit_fixed_topology_variational_approximation
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment

_ADAM_KEY = "adam"
optimizer_classes = {_ADAM_KEY: keras_optimizers.Adam}

_JC_KEY = "jc"
substitution_model_classes = {_JC_KEY: substitution.JC}

_FIXED_STRICT = "fixed-strict"
_CONSTANT_COALESCENT = "constant-coalescent"


def parse_init_values(init_values_string: str) -> tp.Dict[str, tf.Tensor]:
    str_dict = dict(item.split("=") for item in init_values_string.split(","))
    return {key: float(value) for key, value in str_dict.items()}


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
    "-s",
    "--substitution-model",
    required=True,
    type=click.Choice(
        list(substitution_model_classes.keys()),
    ),
    default=_JC_KEY,
)
@click.option(
    "-c",
    "--clock-model",
    required=True,
    type=click.Choice(
        [_FIXED_STRICT],
    ),
    default=_FIXED_STRICT,
)
@click.option(
    "-p",
    "--tree-prior",
    required=True,
    type=click.Choice(
        [_CONSTANT_COALESCENT],
    ),
    default=_CONSTANT_COALESCENT,
)
@click.option(
    "--init-values",
    required=True,
    type=str,
)
@click.option("--approx-output", required=False, type=click.Path())
@click.option("--trace-output", required=False, type=click.Path())
@click.option("-r", "--learning-rate", required=True, type=float, default=1e-3)
def treeflow_vi(
    input,
    topology,
    num_steps,
    optimizer,
    substitution_model,
    clock_model,
    tree_prior,
    learning_rate,
    init_values,
    approx_output,
    trace_output,
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

    if (
        clock_model == _FIXED_STRICT
        and substitution_model == _JC_KEY
        and tree_prior == _CONSTANT_COALESCENT
    ):
        model = get_example_phylo_model(
            taxon_count=tree.taxon_count,
            site_count=alignment.site_count,
            sampling_times=tree.sampling_times,
            pattern_counts=pattern_counts,
            init_values=init_values_dict,
        )
    else:
        raise ValueError(
            "Only example with JC, fixed-rate strict clock, constant coalescent supported for now"
        )
    pinned_model = model.experimental_pin(alignment=encoded_sequences)
    model_names = set(pinned_model._flat_resolve_names())

    init_loc = {
        key: value for key, value in init_values_dict.items() if key in model_names
    }
    init_loc["tree"] = tree

    print(f"Running VI for {num_steps} iterations...")
    approx, trace = fit_fixed_topology_variational_approximation(
        model=pinned_model,
        topologies={DEFAULT_TREE_NAME: tree.topology},
        init_loc=init_loc,
        optimizer=optimizer,
        num_steps=num_steps,
    )
    print("Inference complete")
    print("Approx sample:")
    print(tf.nest.map_structure(lambda x: x.numpy(), approx.sample()))

    if approx_output is not None:
        print(f"Saving approximation to {approx_output}...")
        with open(approx_output, "wb") as f:
            pickle.dump(approx, f)  # TODO: Support saving approximation

    if trace_output is not None:
        print(f"Saving trace to {trace_output}...")
        with open(trace_output, "wb") as f:
            pickle.dump(trace, f)

    print("Exiting...")
