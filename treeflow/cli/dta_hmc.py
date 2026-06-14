"""CLI entry point for discrete-trait (phylogeography / DTA) HMC/NUTS on a
fixed time-tree.

Parallels ``treeflow_hmc`` but loads a two-column CSV of tip traits instead
of an alignment.
"""
from __future__ import annotations

import click
import tensorflow as tf
import yaml

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.cli.inference_common import (
    InitialValueParseError,
    get_tree_vars,
    parse_init_values,
    write_trees,
)
from treeflow.evolution.traitio import DiscreteTraitData, TraitDataParseError
from treeflow.model.io import write_samples_to_file
from treeflow.model.phylo_model import (
    DEFAULT_TREE_VAR_NAME,
    DISCRETE_TRAIT_KEY,
    PhyloModel,
    PhyloModelParseError,
    phylo_model_to_joint_distribution,
)
from treeflow.tree.io import TreeParseError, parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.vi.hmc import KERNEL_HMC, KERNEL_NUTS, fit_fixed_topology_hmc

KERNEL_CHOICES = [KERNEL_HMC, KERNEL_NUTS]

DEFAULT_TAXON_COLUMN = "taxon"
DEFAULT_TRAIT_COLUMN = "trait"


@click.command()
@click.option(
    "-r",
    "--traits",
    required=True,
    type=click.Path(exists=True),
    help="CSV file of discrete tip traits (two columns: taxon and trait).",
)
@click.option(
    "-t",
    "--topology",
    required=True,
    type=click.Path(exists=True),
    help="Fixed Newick tree file (branch lengths in time units).",
)
@click.option(
    "-m",
    "--model-file",
    required=True,
    type=click.Path(exists=True),
    help="YAML model definition file (must contain a `discrete_trait` "
    "substitution block).",
)
@click.option(
    "-n",
    "--num-results",
    type=int,
    default=1000,
    show_default=True,
    help="Number of MCMC samples to collect after burn-in",
)
@click.option(
    "--num-burnin-steps",
    type=int,
    default=500,
    show_default=True,
    help="Number of burn-in steps (discarded)",
)
@click.option(
    "--num-adaptation-steps",
    type=int,
    default=None,
    help="Steps for dual-averaging step-size adaptation "
    "(default: num-burnin-steps)",
)
@click.option(
    "--step-size",
    type=float,
    default=0.01,
    show_default=True,
    help="Initial leapfrog step size",
)
@click.option(
    "--num-leapfrog-steps",
    type=int,
    default=10,
    show_default=True,
    help="Number of leapfrog steps per HMC proposal (ignored for NUTS)",
)
@click.option(
    "--kernel",
    type=click.Choice(KERNEL_CHOICES),
    default=KERNEL_NUTS,
    show_default=True,
    help="MCMC kernel type",
)
@click.option(
    "--init-values",
    required=False,
    type=str,
    help="Initial values: 'scalar=v1,vector=v2a|v2b'",
)
@click.option("-s", "--seed", required=False, type=int)
@click.option(
    "--samples-output",
    required=False,
    type=click.Path(),
    help="Path to save parameter samples (CSV).",
)
@click.option(
    "--taxon-column",
    type=str,
    default=DEFAULT_TAXON_COLUMN,
    show_default=True,
    help="Name of the column in the traits CSV holding taxon labels.",
)
@click.option(
    "--trait-column",
    type=str,
    default=DEFAULT_TRAIT_COLUMN,
    show_default=True,
    help="Name of the column in the traits CSV holding discrete trait labels.",
)
@click.option(
    "--states",
    type=str,
    default=None,
    help="Comma-separated explicit ordering of trait states "
    "(defaults to sorted unique labels observed in the CSV).",
)
@click.option(
    "--subnewick-format",
    type=int,
    default=0,
    show_default=True,
    help="Subnewick format (see `ete3.Tree`)",
)
def treeflow_dta_hmc(
    traits,
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
    taxon_column,
    trait_column,
    states,
    subnewick_format,
):
    """Fixed-topology HMC/NUTS inference for a discrete-trait (DTA /
    phylogeography) model on a fixed time-tree.

    Given a Newick tree with branch lengths in time units and a CSV of tip
    trait labels, runs NUTS (default) or HMC to sample the posterior over
    exchange rates, equilibrium frequencies, and any other free parameters
    declared in the YAML model file.
    """
    print(f"Parsing topology {topology}")
    try:
        tree = convert_tree_to_tensor(
            parse_newick(topology, subnewick_format=subnewick_format)
        )
    except TreeParseError as ex:
        raise click.ClickException(str(ex))

    print(f"Parsing traits {traits}")
    state_tuple = None if states is None else tuple(
        s.strip() for s in states.split(",")
    )
    try:
        trait_data = DiscreteTraitData(
            csv_file=traits,
            states=state_tuple,
            taxon_column=taxon_column,
            trait_column=trait_column,
        )
    except TraitDataParseError as ex:
        raise click.ClickException(str(ex))

    tree_taxa = list(tree.taxon_set)
    missing = set(tree_taxa) - set(trait_data.trait_mapping.keys())
    if missing:
        raise click.ClickException(
            f"Traits CSV is missing {len(missing)} taxa present in the "
            f"tree: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    encoded_traits = trait_data.get_encoded_trait_tensor(tree_taxa)
    print(
        f"  {trait_data.taxon_count} tips, {trait_data.n_states} states: "
        f"{list(trait_data.states)}"
    )

    print("Parsing model...")
    with open(model_file) as f:
        model_dict = yaml.safe_load(f)
    try:
        phylo_model = PhyloModel(model_dict)
    except PhyloModelParseError as ex:
        raise click.ClickException(str(ex))

    if phylo_model.subst_model != DISCRETE_TRAIT_KEY:
        raise click.ClickException(
            f"Expected substitution model {DISCRETE_TRAIT_KEY!r} in model "
            f"file, got {phylo_model.subst_model!r}."
        )
    declared_n_states = int(phylo_model.subst_params["n_states"])
    if declared_n_states != trait_data.n_states:
        raise click.ClickException(
            f"Model declares n_states={declared_n_states} but the traits "
            f"CSV contains {trait_data.n_states} observed states "
            f"({list(trait_data.states)}). Either fix the model or pass "
            f"--states to include unobserved states in the state space."
        )

    model = phylo_model_to_joint_distribution(phylo_model, tree, trait_data)
    pinned_model = model.experimental_pin(alignment=encoded_traits)
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
        f"Running {kernel}: {num_burnin_steps} burn-in + {num_results} "
        f"sampling steps ({total_steps} total)..."
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

    if samples_output is not None:
        constrained_samples = hmc_res.samples
        names = pinned_model._flat_resolve_names()
        flat_samples = pinned_model._model_flatten(constrained_samples)
        samples_dict = dict(zip(names, flat_samples))

        tree_vars = get_tree_vars(phylo_model)
        tree_samples = {
            var: samples_dict.pop(var) for var in tree_vars if var in samples_dict
        }

        print(f"Saving samples to {samples_output}...")
        write_samples_to_file(
            constrained_samples,
            pinned_model,
            samples_output,
            vars=samples_dict.keys(),
            tree_vars=(
                {DEFAULT_TREE_VAR_NAME: tree_samples[DEFAULT_TREE_VAR_NAME]}
                if DEFAULT_TREE_VAR_NAME in tree_samples
                else None
            ),
        )

    print("Exiting...")
