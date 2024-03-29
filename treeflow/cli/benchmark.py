import click
from timeit import default_timer as timer
import io
import typing as tp
import tensorflow as tf
import numpy as np
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.evolution.substitution.nucleotide.jc import JC
from treeflow.evolution.substitution.nucleotide.gtr import GTR
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)
from treeflow.distributions.sample_weighted import SampleWeighted
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.traversal.anchor_heights import get_anchor_heights
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from itertools import product

T = tp.TypeVar("T")


def time_fn(
    f: tp.Callable[..., T]
) -> tp.Callable[[int, tp.Iterable[tf.Tensor]], tp.Tuple[float, T]]:
    def timed(replicates: int, args: tp.Iterable[tf.Tensor]):
        start = timer()
        for _ in range(replicates):
            out = f(*args)
        end = timer()
        total_time = end - start
        return total_time, out

    return timed


def get_tree_likelihood_computation(
    tree: TensorflowRootedTree,
    input: Alignment,
    dtype: tf.DType,
    bito_instance: tp.Optional[object] = None,
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], tf.Tensor]:
    base_unrooted_tree = tree.get_unrooted_tree()
    subst_model = JC()
    if bito_instance is not None:
        from treeflow.acceleration.bito.beagle import (
            phylogenetic_likelihood as beagle_likelihood,
        )

        log_prob, _ = beagle_likelihood(
            input.fasta_file,
            subst_model,
            subst_model.frequencies(dtype=dtype),
            inst=bito_instance,
        )
    else:
        compressed_alignment = input.get_compressed_alignment()
        encoded_sequences = compressed_alignment.get_encoded_sequence_tensor(
            tree.taxon_set, dtype=dtype
        )
        weights = compressed_alignment.get_weights_tensor(dtype=dtype)
        site_count = compressed_alignment.site_count
        frequencies = subst_model.frequencies(dtype=dtype)

        def log_prob(branch_lengths: tf.Tensor):
            tree = base_unrooted_tree.with_branch_lengths(branch_lengths)
            transition_probs_tree = get_transition_probabilities_tree(
                tree, subst_model, dtype=dtype
            )
            dist = SampleWeighted(
                LeafCTMC(transition_probs_tree, frequencies),
                weights=weights,
                sample_shape=(site_count,),
            )
            return dist.log_prob(encoded_sequences)

    return log_prob, base_unrooted_tree.branch_lengths


def get_tree_likelihood_gtr_computation(
    tree: TensorflowRootedTree,
    input: Alignment,
    dtype: tf.DType,
    bito_instance: tp.Optional[object] = None,
):
    base_unrooted_tree = tree.get_unrooted_tree()
    subst_model = GTR()
    init_frequencies = tf.constant([0.21, 0.28, 0.27, 0.24], dtype=dtype)
    init_rates = tf.constant([0.2, 0.12, 0.17, 0.09, 0.24, 0.18], dtype=dtype)
    if bito_instance is not None:
        raise NotImplemented("GTR gradients with bito not yet implemented")
    else:
        compressed_alignment = input.get_compressed_alignment()
        encoded_sequences = compressed_alignment.get_encoded_sequence_tensor(
            tree.taxon_set, dtype=dtype
        )
        weights = compressed_alignment.get_weights_tensor(dtype=dtype)
        site_count = compressed_alignment.site_count

        def log_prob(
            branch_lengths: tf.Tensor, rates: tf.Tensor, frequencies: tf.Tensor
        ):
            tree = base_unrooted_tree.with_branch_lengths(branch_lengths)
            transition_probs_tree = get_transition_probabilities_tree(
                tree, subst_model, rates=rates, frequencies=frequencies
            )
            dist = SampleWeighted(
                LeafCTMC(transition_probs_tree, frequencies),
                weights=weights,
                sample_shape=(site_count,),
            )
            return dist.log_prob(encoded_sequences)

    return log_prob, base_unrooted_tree.branch_lengths, init_rates, init_frequencies


def get_ratio_transform_computation(
    tree: TensorflowRootedTree,
    dtype: tf.DType,
    bito_instance: tp.Optional[object] = None,
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], tf.Tensor]:

    anchor_heights = tf.constant(get_anchor_heights(tree.numpy()), dtype=dtype)
    init_bijector = NodeHeightRatioBijector(tree.topology, anchor_heights)
    init_ratios = tf.constant(init_bijector.inverse(tree.node_heights))

    if bito_instance is not None:
        from treeflow.acceleration.bito.ratio_transform import ratios_to_node_heights

        def ratio_transform_forward(ratios):
            return ratios_to_node_heights(bito_instance, anchor_heights, ratios)

    else:

        def ratio_transform_forward(ratios):
            bijector = NodeHeightRatioBijector(
                tree.topology, anchor_heights
            )  # Avoid caching
            return bijector.forward(ratios)

    return ratio_transform_forward, init_ratios


def get_ratio_transform_jacobian_computation(
    tree: TensorflowRootedTree,
    dtype: tf.DType,
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], tf.Tensor]:

    anchor_heights = tf.constant(get_anchor_heights(tree.numpy()), dtype=dtype)
    init_node_heights = tree.node_heights

    def ratio_transform_forward(node_heights):
        bijector = NodeHeightRatioBijector(
            tree.topology, anchor_heights
        )  # Avoid caching
        return -bijector.inverse_log_det_jacobian(node_heights)

    return ratio_transform_forward, init_node_heights


def get_constant_coalescent_computation(
    base_tree: TensorflowRootedTree, dtype: tf.DType
) -> tp.Tuple[
    tp.Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    tp.Tuple[tf.Tensor, tf.Tensor],
]:
    pop_size = tf.constant(4.0, dtype=dtype)

    def coalescent_log_prob(pop_size: tf.Tensor, node_heights: tf.Tensor):
        dist = ConstantCoalescent(
            base_tree.taxon_count, pop_size, base_tree.sampling_times
        )
        tree = base_tree.with_node_heights(node_heights)
        return dist.log_prob(tree)

    return coalescent_log_prob, (pop_size, base_tree.node_heights)


def get_gradient_fn(
    task_fn: tp.Callable[..., tf.Tensor],
    output_gradients: tp.Optional[tf.Tensor] = None,
) -> tp.Callable[..., tf.Tensor]:
    def gradient(*args: tp.Iterable[tf.Tensor]):
        with tf.GradientTape() as t:
            for arg in args:
                t.watch(arg)
            res = task_fn(*args)
        gradient_res = t.gradient(res, args, output_gradients=output_gradients)
        return res, gradient_res

    return gradient


def benchmark(
    tree: TensorflowRootedTree,
    input: Alignment,
    dtype: tf.DType,
    scaler: tf.Tensor,
    computation: str,
    task: str,
    jit: bool,
    precompile: bool,
    replicates: int,
    bito_instance: tp.Optional[object] = None,
) -> tp.Tuple[float, tp.Optional[float]]:

    task_fn: tp.Callable[..., tf.Tensor]
    args: tp.Iterable[tf.Tensor]
    output_gradients: tp.Optional[tf.Tensor] = None
    if computation == "treelikelihood":
        (task_fn, branch_lengths,) = get_tree_likelihood_computation(
            tree, input, dtype, bito_instance=bito_instance
        )
        args = (branch_lengths * scaler,)
    elif computation == "treelikelihoodGTR":
        (
            task_fn,
            branch_lengths,
            rates,
            frequencies,
        ) = get_tree_likelihood_gtr_computation(
            tree, input, dtype, bito_instance=bito_instance
        )
        args = (branch_lengths * scaler, rates, frequencies)
    elif computation == "ratio_transform":
        task_fn, ratios = get_ratio_transform_computation(
            tree, dtype, bito_instance=bito_instance
        )
        output_gradients = tf.ones_like(ratios)
        args = (ratios,)
    elif computation == "ratio_transform_jacobian":
        task_fn, node_heights = get_ratio_transform_jacobian_computation(tree, dtype)
        args = (node_heights,)
    elif computation == "constant_coalescent":
        task_fn, args = get_constant_coalescent_computation(tree, dtype=dtype)
    else:
        raise ValueError(f"Unknown computation: {computation}")

    fn: tp.Callable[..., tf.Tensor]
    if task == "gradient":
        fn = get_gradient_fn(task_fn, output_gradients=output_gradients)
    elif task == "evaluation":
        fn = task_fn
    else:
        raise ValueError(f"Unknown task: {task}")

    if jit:
        fn = tf.function(fn)
    if precompile:
        print("Precompiling...")
        fn(*args)

    print("Running benchmark...")
    timed_fn = time_fn(fn)
    time, raw_value = timed_fn(replicates, args)

    if task == "gradient":
        value, _ = raw_value
    else:
        value = raw_value

    squeezed = tf.squeeze(value)
    numpy_value = squeezed.numpy() if squeezed.shape.rank == 0 else None

    print(f"Value: {numpy_value if numpy_value else 'nonscalar'}")
    print(f"Time: {time}")
    return time, numpy_value


DTYPE_MAPPING = dict(float32=tf.float32, float64=tf.float64)


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Alignment file (FASTA format)",
)
@click.option(
    "-t", "--tree", required=True, type=click.Path(exists=True), help="Tree file"
)
@click.option(
    "-r", "--replicates", required=True, type=int, help="Number of replicates"
)
@click.option(
    "-o", "--output", type=click.File(mode="w"), help="Output file (CSV format)"
)
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(["float32", "float64"]),
    default="float64",
    help="Floating point data type",
)
@click.option(
    "-p",
    "--precompile",
    type=bool,
    default=True,
    help="Don't include function compilation time",
)
@click.option(
    "-s",
    "--scaler",
    type=float,
    default=1.0,
    help="Scale branch lengths",
)
@click.option(
    "-e",
    "--eager",
    type=bool,
    default=False,
    is_flag=True,
    help="Include eager mode in benchmark",
)
@click.option(
    "-m",
    "--memory",
    type=bool,
    default=False,
    is_flag=True,
    help="Profile maximum memory usage in benchmark",
)
@click.option(
    "-g",
    "--gtr",
    type=bool,
    default=False,
    is_flag=True,
    help="Include GTR likelihood in benchmark",
)
@click.option("--use-bito", is_flag=True)
def treeflow_benchmark(
    input: str,
    tree: str,
    replicates: int,
    output: tp.Optional[io.StringIO],
    dtype: str,
    scaler: float,
    precompile: bool,
    eager: bool,
    memory: bool,
    use_bito: bool,
    gtr: bool,
):

    print("Parsing input...")
    alignment = Alignment(input)
    numpy_tree = parse_newick(tree, remove_zero_edges=True)
    tf_dtype = DTYPE_MAPPING[dtype]
    tensor_tree = convert_tree_to_tensor(numpy_tree, height_dtype=tf_dtype)
    scaler_tensor = tf.constant(scaler, dtype=tf_dtype)

    computations = [
        "treelikelihood",
        "ratio_transform",
        "ratio_transform_jacobian",
        "constant_coalescent",
    ]
    if gtr:
        computations += ["treelikelihoodGTR"]
    tasks = ["gradient", "evaluation"]
    jits = [True, False] if eager else [True]

    if output:
        output.write("function,mode,JIT,time,logprob")
        if memory:
            output.write(",max_mem")
        output.write("\n")

    print("Starting benchmark...")
    if use_bito:
        from treeflow.acceleration.bito.instance import get_instance

        dated = not np.allclose(numpy_tree.sampling_times, 0.0)
        bito_instance = get_instance(tree, dated=dated)
    else:
        bito_instance = None
    for (computation, task, jit) in product(computations, tasks, jits):
        print(
            f"Benchmarking {computation} {task}{' in function mode' if jit else ''}..."
        )
        benchmark_args = (
            tensor_tree,
            alignment,
            tf_dtype,
            scaler_tensor,
            computation,
            task,
            jit,
            precompile,
            replicates,
        )
        max_mem: tp.Optional[float]
        if memory:
            from memory_profiler import memory_usage

            max_mem, (time, value) = memory_usage(
                (benchmark, benchmark_args, dict(bito_instance=bito_instance)),
                retval=True,
                max_usage=True,
                max_iterations=1,
            )
        else:
            time, value = benchmark(*benchmark_args, bito_instance=bito_instance)
            max_mem = None
        if output:
            jit_str = "on" if jit else "off"
            output.write(
                f"{computation},{task},{jit_str},{time},{value if value else ''}"
            )
            if max_mem is not None:
                output.write(f",{max_mem}")
            output.write("\n")
        if max_mem is not None:
            print(f"Max memory usage: {max_mem}")
        print("\n")
    print("Benchmark complete")
