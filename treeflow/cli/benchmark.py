import click
from timeit import default_timer as timer
import io
import typing as tp
import numpy as np
import tensorflow as tf
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment, WeightedAlignment
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.evolution.substitution.nucleotide.jc import JC
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
    tree: TensorflowRootedTree, input: Alignment, dtype: tf.DType
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], tf.Tensor]:
    compressed_alignment = input.get_compressed_alignment()
    encoded_sequences = compressed_alignment.get_encoded_sequence_tensor(
        tree.taxon_set, dtype=dtype
    )
    weights = compressed_alignment.get_weights_tensor(dtype=dtype)
    site_count = compressed_alignment.site_count
    subst_model = JC()
    frequencies = subst_model.frequencies(dtype=dtype)
    base_unrooted_tree = tree.get_unrooted_tree()

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


def get_ratio_transform_jacobian_computation(
    tree: TensorflowRootedTree, dtype: tf.DType
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], tf.Tensor]:

    anchor_heights = tf.constant(get_anchor_heights(tree.numpy()), dtype=dtype)
    init_bijector = NodeHeightRatioBijector(tree.topology, anchor_heights)
    ratios = tf.constant(
        init_bijector.inverse(tree.internal_node_heights).numpy(), dtype=dtype
    )

    def log_det_jacobian(ratios):
        bijector = NodeHeightRatioBijector(
            tree.topology, anchor_heights
        )  # Avoid caching
        return bijector.forward_log_det_jacobian(ratios)

    return log_det_jacobian, ratios


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
        tree = base_tree.with_internal_node_heights(node_heights)
        return dist.log_prob(tree)

    return coalescent_log_prob, (pop_size, base_tree.internal_node_heights)


def get_gradient_fn(
    task_fn: tp.Callable[..., tf.Tensor]
) -> tp.Callable[..., tp.Iterable[tf.Tensor]]:
    def gradient(*args: tp.Iterable[tf.Tensor]):
        with tf.GradientTape() as t:
            for arg in args:
                t.watch(arg)
            res = task_fn(*args)
        gradient_res = t.gradient(res, args)
        return gradient_res

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
) -> tp.Tuple[float, tf.Tensor]:

    task_fn: tp.Callable[..., tf.Tensor]
    args: tp.Iterable[tf.Tensor]
    if computation == "treelikelihood":
        (
            task_fn,
            branch_lengths,
        ) = get_tree_likelihood_computation(tree, input, dtype)
        args = (branch_lengths * scaler,)
    elif computation == "ratio_transform_jacobian":
        task_fn, ratios = get_ratio_transform_jacobian_computation(tree, dtype)
        args = (ratios,)
    elif computation == "constant_coalescent":
        task_fn, args = get_constant_coalescent_computation(tree, dtype=dtype)
    else:
        raise ValueError(f"Unknown computation: {computation}")

    fn: tp.Callable[[tp.Iterable[tf.Tensor]], tp.Iterable[tf.Tensor]]
    if task == "gradient":
        fn = get_gradient_fn(task_fn)
    elif task == "evaluation":
        fn = lambda *args: (task_fn(*args),)
    else:
        raise ValueError(f"Unknown task: {task}")

    if jit:
        fn = tf.function(fn)
    if precompile:
        print("Precompiling...")
        fn(*args)

    print("Running benchmark...")
    timed_fn = time_fn(fn)
    time, value = timed_fn(replicates, args)

    print(f"Value: {[element.numpy() for element in value]}")
    print(f"Time: {time}")
    return time, value


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
def treeflow_benchmark(
    input: str,
    tree: str,
    replicates: int,
    output: tp.Optional[io.StringIO],
    dtype: str,
    scaler: float,
    precompile: bool,
):
    print("Parsing input...")
    alignment = Alignment(input)
    numpy_tree = parse_newick(tree)
    tf_dtype = DTYPE_MAPPING[dtype]
    tensor_tree = convert_tree_to_tensor(numpy_tree, height_dtype=tf_dtype)
    scaler_tensor = tf.constant(scaler, dtype=tf_dtype)

    computations = ["treelikelihood", "ratio_transform_jacobian", "constant_coalescent"]
    tasks = ["gradient", "evaluation"]
    jits = [True, False]

    if output:
        output.write("function,mode,JIT,time,logprob\n")

    print("Starting benchmark...")
    for (computation, task, jit) in product(computations, tasks, jits):
        print(
            f"Benchmarking {computation} {task}{' in function mode' if jit else ''}..."
        )
        time, value = benchmark(
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
        print("\n")
        np_value = [np.squeeze(element.numpy()) for element in value]
        if output:
            output.write(f"{computation},{task},{jit},{time},{np_value}")
    print("Benchmark complete")
