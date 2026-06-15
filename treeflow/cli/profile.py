"""Profiling CLI for the per-iteration cost of phylogenetic inference.

A single ADVI/HMC iteration is dominated by one evaluation of the model's
unnormalised log density and its gradient (the reparameterised ELBO gradient
for ADVI, the leapfrog gradient for HMC). This command measures that evaluation
and breaks it down into the contributions that matter for performance work:

* ``likelihood``  -- the tree (Felsenstein) likelihood subsystem, including the
  per-branch transition-probability construction. This is the term accelerated
  by the native C++ op, so it is measured for each requested engine.
* ``prior``       -- the tree prior plus all parameter priors.
* ``overhead``    -- the fixed per-step cost of assembling the tree from node
  heights, structure (un)packing and automatic-differentiation bookkeeping.
* ``ratio_transform`` -- the node-height ratio bijector (forward transform plus
  log-det-Jacobian) that maps the unconstrained tree parameters used by
  inference to node heights. This is part of the reparameterisation, separate
  from the density evaluation, and is reported alongside it.

The three density components are additive by construction:
``likelihood + prior + overhead == full``. They are obtained as nested marginal
differences of value-and-gradient timings so that the shared per-call overhead
is attributed once, to ``overhead``, rather than double counted.

To demonstrate scaling, the command runs over a range of synthetic datasets of
increasing taxon count (coalescent trees with random alignments; the sequence
content does not affect timing). A real alignment/tree pair can also be supplied.
"""

from __future__ import annotations

import io
import os
import random
import timeit
import typing as tp

import click
import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.evolution.seqio import Alignment
from treeflow.model.phylo_model import (
    PhyloModel,
    phylo_model_to_joint_distribution,
    DEFAULT_TREE_VAR_NAME,
)
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.traversal.anchor_heights import get_anchor_heights
from treeflow.cli.inference_common import EXAMPLE_PHYLO_MODEL_DICT
from treeflow.acceleration.native import is_available as native_is_available
from treeflow.acceleration.native import (
    ratio_transform_is_available as native_ratio_is_available,
)


DTYPES = {"float32": tf.float32, "float64": tf.float64}

# Components of the density evaluation, in the order they are reported. These
# sum to the full target evaluation.
DENSITY_COMPONENTS = ["likelihood", "prior", "overhead"]

# How each engine evaluates the tree traversals (likelihood + ratio transform):
#   native  -- the hand-written C++ ops
#   static  -- pure TensorFlow, traversal unrolled into a straight-line graph for
#              the (statically known) topology
#   dynamic -- pure TensorFlow, traversal as a tf.while_loop over a TensorArray
#              (handles a topology that is not known at trace time)
ENGINE_CONFIGS = {
    "native": dict(use_native=True, unroll="auto"),
    "static": dict(use_native=False, unroll=True),
    "dynamic": dict(use_native=False, unroll=False),
}
ENGINE_ALIASES = {"tf": "dynamic"}  # back-compat: "tf" == the dynamic TF engine


class Dataset(tp.NamedTuple):
    name: str
    tree: TensorflowRootedTree
    alignment: Alignment
    encoded_sequences: tf.Tensor
    pattern_counts: tf.Tensor

    @property
    def taxon_count(self) -> int:
        return int(self.tree.taxon_count)

    @property
    def pattern_count(self) -> int:
        return int(self.pattern_counts.shape[0])


def _dataset_from_files(
    name: str, tree_path: str, fasta_path: str, dtype: tf.DType
) -> Dataset:
    tree = convert_tree_to_tensor(parse_newick(tree_path), height_dtype=dtype)
    alignment = Alignment(fasta_path).get_compressed_alignment()
    encoded = alignment.get_encoded_sequence_tensor(tree.taxon_set)
    pattern_counts = alignment.get_weights_tensor()
    return Dataset(name, tree, alignment, encoded, pattern_counts)


def simulate_dataset(
    n_taxa: int, n_sites: int, seed: int, dtype: tf.DType, tmp_dir: str
) -> Dataset:
    """Simulate a coalescent tree and a random alignment of the given size.

    The alignment content is random: the tree-likelihood cost depends on the
    number of taxa, sites and states, not on the nucleotides themselves, so this
    is sufficient (and reproducible) for benchmarking.
    """
    import dendropy

    rng = np.random.default_rng(seed)
    taxa = dendropy.TaxonNamespace([f"T{i}" for i in range(n_taxa)])
    dtree = dendropy.simulate.treesim.pure_kingman_tree(
        taxon_namespace=taxa, pop_size=1.0, rng=random.Random(seed)
    )
    newick = dtree.as_string(schema="newick", suppress_rooting=True)

    bases = np.array(list("ACGT"))
    fasta = io.StringIO()
    for taxon in taxa:
        sequence = "".join(rng.choice(bases, size=n_sites))
        fasta.write(f">{taxon.label}\n{sequence}\n")

    tree_path = os.path.join(tmp_dir, f"sim_{n_taxa}.nwk")
    fasta_path = os.path.join(tmp_dir, f"sim_{n_taxa}.fasta")
    with open(tree_path, "w") as f:
        f.write(newick)
    with open(fasta_path, "w") as f:
        f.write(fasta.getvalue())

    return _dataset_from_files(f"sim-{n_taxa}taxa", tree_path, fasta_path, dtype)


def time_value_and_grad(
    fn: tp.Callable[..., tf.Tensor],
    args: tp.Sequence[tf.Tensor],
    replicates: int,
) -> float:
    """Mean wall-clock seconds for one compiled value-and-gradient evaluation.

    The function is wrapped in ``tf.function`` and traced once before timing, so
    graph-construction time is excluded -- this reflects the steady state inside
    an inference loop running under ``tf.function``.
    """

    @tf.function
    def value_and_grad(*tensors):
        with tf.GradientTape() as tape:
            for tensor in tensors:
                tape.watch(tensor)
            value = fn(*tensors)
        return value, tape.gradient(value, tensors)

    def force(value, grads):
        value.numpy()
        for g in grads:
            if g is not None:
                g.numpy()

    force(*value_and_grad(*args))  # trace + warm up
    start = timeit.default_timer()
    for _ in range(replicates):
        result = value_and_grad(*args)
    force(*result)
    return (timeit.default_timer() - start) / replicates


def _find_tree_field(sample) -> str:
    for name, value in sample._asdict().items():
        if isinstance(value, TensorflowRootedTree):
            return name
    raise ValueError("Model sample contains no tree component")


def profile_dataset(
    dataset: Dataset,
    phylo_model: PhyloModel,
    engines: tp.Sequence[str],
    replicates: int,
    dtype: tf.DType,
    seed: int,
) -> tp.List[dict]:
    """Time the density components and ratio transform for one dataset."""
    tree = dataset.tree
    enc = dataset.encoded_sequences
    pc = dataset.pattern_counts

    full_models = {
        engine: phylo_model_to_joint_distribution(
            phylo_model,
            tree,
            dataset.alignment,
            pattern_counts=pc,
            **ENGINE_CONFIGS[engine],
        )
        for engine in engines
    }
    prior_model = phylo_model_to_joint_distribution(
        phylo_model, tree, dataset.alignment, pattern_counts=pc, include_likelihood=False
    )
    pinned = {
        engine: model.experimental_pin(alignment=enc)
        for engine, model in full_models.items()
    }

    tf_seed = tf.constant([seed, seed + 1], dtype=tf.int32)
    sample = full_models[engines[0]].sample(seed=tf_seed)
    sample_dict = sample._asdict()
    tree_field = _find_tree_field(sample)
    tree_obj = sample_dict[tree_field]
    node_heights = tf.identity(tree_obj.node_heights)
    other_latents = {
        name: tf.identity(value)
        for name, value in sample_dict.items()
        if name not in (tree_field, "alignment")
    }
    other_keys = list(other_latents)

    def assemble(heights, *other_values) -> tp.Dict[str, object]:
        latents = {tree_field: tree_obj.with_node_heights(heights)}
        latents.update(dict(zip(other_keys, other_values)))
        return latents

    grad_args = [node_heights] + [other_latents[k] for k in other_keys]

    # Full target (prior + likelihood) per engine.
    full_time = {
        engine: time_value_and_grad(
            lambda h, *o, e=engine: pinned[e].unnormalized_log_prob(**assemble(h, *o)),
            grad_args,
            replicates,
        )
        for engine in engines
    }

    # Prior-only target (engine independent).
    prior_time = time_value_and_grad(
        lambda h, *o: prior_model.log_prob(**assemble(h, *o)),
        grad_args,
        replicates,
    )

    # Fixed per-step overhead: assemble the tree from heights and touch every
    # latent so that tree reconstruction and gradient bookkeeping are timed, but
    # no density is evaluated.
    def overhead_fn(heights, *other_values):
        latents = assemble(heights, *other_values)
        return tf.add_n(
            [tf.reduce_sum(latents[tree_field].node_heights)]
            + [tf.reduce_sum(v) for v in other_values]
        )

    overhead_time = time_value_and_grad(overhead_fn, grad_args, replicates)

    # Node-height ratio transform (reparameterisation). It is part of the same
    # static/dynamic/native traversal story as the likelihood, so time it for each
    # requested engine (skipping native if that specific op is not built).
    anchor_heights = tf.constant(get_anchor_heights(tree.numpy()), dtype=dtype)
    ratios = tf.identity(
        NodeHeightRatioBijector(
            tree.topology, anchor_heights, use_native=False, unroll=False
        ).inverse(node_heights)
    )

    def make_ratio_fn(bijector):
        def ratio_fn(r):
            return tf.reduce_sum(
                bijector.forward(r)
            ) + bijector.forward_log_det_jacobian(r, event_ndims=1)

        return ratio_fn

    ratio_times: tp.Dict[str, float] = {}
    for engine in engines:
        if engine == "native" and not native_ratio_is_available():
            continue
        cfg = ENGINE_CONFIGS[engine]
        bijector = NodeHeightRatioBijector(
            tree.topology,
            anchor_heights,
            use_native=cfg["use_native"],
            unroll=cfg["unroll"],
        )
        ratio_times[engine] = time_value_and_grad(
            make_ratio_fn(bijector), [ratios], replicates
        )

    rows: tp.List[dict] = []
    base = dict(
        dataset=dataset.name,
        taxa=dataset.taxon_count,
        patterns=dataset.pattern_count,
        sites=int(dataset.alignment.site_count),
    )
    for engine in engines:
        likelihood = full_time[engine] - prior_time
        prior = prior_time - overhead_time
        component_times = {
            "likelihood": likelihood,
            "prior": prior,
            "overhead": overhead_time,
            "full": full_time[engine],
        }
        for component, seconds in component_times.items():
            rows.append(
                dict(base, engine=engine, component=component, time_ms=seconds * 1e3)
            )
    # Ratio transform, per engine (static/dynamic/native).
    for engine, seconds in ratio_times.items():
        rows.append(
            dict(
                base,
                engine=engine,
                component="ratio_transform",
                time_ms=seconds * 1e3,
            )
        )
    return rows


def _print_dataset_summary(dataset: Dataset, rows: tp.List[dict], engines):
    print(
        f"\n{dataset.name}: {dataset.taxon_count} taxa, "
        f"{int(dataset.alignment.site_count)} sites, "
        f"{dataset.pattern_count} patterns"
    )

    def lookup(engine, component):
        for row in rows:
            if (
                row["dataset"] == dataset.name
                and row["engine"] == engine
                and row["component"] == component
            ):
                return row["time_ms"]
        return float("nan")

    header = f"  {'component':<16}" + "".join(f"{e:>12}" for e in engines)
    print(header)
    for component in ["likelihood", "prior", "overhead", "full", "ratio_transform"]:
        line = f"  {component:<16}"
        for engine in engines:
            line += f"{lookup(engine, component):>12.3f}"
        print(line)
    # Speedups relative to the dynamic (TensorArray) engine, where present.
    if "dynamic" in engines:
        for engine in engines:
            if engine == "dynamic":
                continue
            for component in ["likelihood", "ratio_transform"]:
                base_t = lookup("dynamic", component)
                eng_t = lookup(engine, component)
                if eng_t and not np.isnan(eng_t) and not np.isnan(base_t):
                    print(
                        f"  -> {engine} {component} speedup vs dynamic: "
                        f"{base_t / eng_t:.1f}x"
                    )


@click.command()
@click.option(
    "--taxa",
    "-T",
    default="16,32,64",
    show_default=True,
    help="Comma-separated taxon counts for synthetic scaling datasets.",
)
@click.option(
    "--sites",
    default=500,
    show_default=True,
    type=int,
    help="Number of alignment sites for synthetic datasets.",
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    help="Optional real alignment (FASTA) to profile in addition to synthetic data.",
)
@click.option(
    "-t",
    "--tree",
    type=click.Path(exists=True),
    help="Tree (Newick) accompanying --input.",
)
@click.option(
    "-m",
    "--model-file",
    type=click.Path(exists=True),
    help="YAML model definition file (defaults to coalescent / strict clock / JC).",
)
@click.option(
    "--engines",
    default="native,static,dynamic",
    show_default=True,
    help=(
        "Comma-separated traversal engines to compare: 'native' (C++ ops), "
        "'static' (pure-TF, unrolled for the static topology) and 'dynamic' "
        "(pure-TF, TensorArray while_loop). 'tf' is accepted as an alias for "
        "'dynamic'."
    ),
)
@click.option(
    "-r",
    "--replicates",
    default=30,
    show_default=True,
    type=int,
    help="Number of timed value-and-gradient evaluations per measurement.",
)
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(list(DTYPES.keys())),
    default="float64",
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Path to write per-component timings as CSV.",
)
@click.option("-s", "--seed", default=1, show_default=True, type=int)
def treeflow_profile(
    taxa,
    sites,
    input,
    tree,
    model_file,
    engines,
    replicates,
    dtype,
    output,
    seed,
):
    """Profile the per-iteration cost of phylogenetic inference and its breakdown
    into likelihood, prior, overhead and ratio-transform components, comparing the
    traversal engines across dataset sizes: ``native`` (C++ ops), ``static`` (pure
    TensorFlow unrolled for the static topology) and ``dynamic`` (pure TensorFlow
    TensorArray while_loop)."""
    import yaml

    tf_dtype = DTYPES[dtype]

    requested_engines = [e.strip() for e in engines.split(",") if e.strip()]
    engine_list = []
    for engine in requested_engines:
        engine = ENGINE_ALIASES.get(engine, engine)
        if engine not in ENGINE_CONFIGS:
            raise click.ClickException(
                f"Unknown engine {engine!r}; choose from "
                f"{', '.join(ENGINE_CONFIGS)} (or alias 'tf')."
            )
        if engine == "native" and not native_is_available():
            click.echo("Native op not available; skipping the 'native' engine.")
            continue
        if engine not in engine_list:
            engine_list.append(engine)
    if not engine_list:
        raise click.ClickException("No usable traversal engines requested.")

    if model_file is None:
        model_dict = EXAMPLE_PHYLO_MODEL_DICT
    else:
        with open(model_file) as f:
            model_dict = yaml.safe_load(f)
    phylo_model = PhyloModel(model_dict)

    taxon_counts = [int(t) for t in taxa.split(",") if t.strip()]

    if bool(input) != bool(tree):
        raise click.ClickException("--input and --tree must be provided together.")

    all_rows: tp.List[dict] = []
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        datasets = [
            simulate_dataset(n, sites, seed, tf_dtype, tmp_dir) for n in taxon_counts
        ]
        if input is not None:
            datasets.append(
                _dataset_from_files("real", tree, input, tf_dtype)
            )

        for dataset in datasets:
            print(f"Profiling {dataset.name} ...")
            rows = profile_dataset(
                dataset, phylo_model, engine_list, replicates, tf_dtype, seed
            )
            all_rows.extend(rows)
            _print_dataset_summary(dataset, rows, engine_list)

    _print_scaling_analysis(all_rows, engine_list)

    if output is not None:
        import pandas as pd

        pd.DataFrame(all_rows).to_csv(output, index=False)
        print(f"\nWrote timings to {output}")


def _print_scaling_analysis(rows: tp.List[dict], engines):
    print("\n=== scaling summary (full target, ms) ===")
    datasets = []
    for row in rows:
        if row["dataset"] not in [d[0] for d in datasets]:
            datasets.append((row["dataset"], row["taxa"]))
    datasets.sort(key=lambda d: d[1])

    def lookup(dataset_name, engine, component):
        for row in rows:
            if (
                row["dataset"] == dataset_name
                and row["engine"] == engine
                and row["component"] == component
            ):
                return row["time_ms"]
        return float("nan")

    # Speedup columns: each non-dynamic engine's likelihood vs the dynamic engine.
    speedup_engines = (
        [e for e in engines if e != "dynamic"] if "dynamic" in engines else []
    )
    header = f"  {'taxa':>6}" + "".join(f"{'full-' + e:>14}" for e in engines)
    header += "".join(f"{e + ' like x':>14}" for e in speedup_engines)
    print(header)
    for name, taxa in datasets:
        line = f"  {taxa:>6}"
        for engine in engines:
            line += f"{lookup(name, engine, 'full'):>14.3f}"
        for engine in speedup_engines:
            base_t = lookup(name, "dynamic", "likelihood")
            eng_t = lookup(name, engine, "likelihood")
            speedup = base_t / eng_t if eng_t else float("nan")
            line += f"{speedup:>13.1f}x"
        print(line)
