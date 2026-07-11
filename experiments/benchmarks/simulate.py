"""Simulate coalescent trees and DNA sequences using treeflow alone.

This replaces the BEAST 2-based simulation used by the old
``treeflow-benchmarks`` pipeline. Two pieces are needed:

* a tree topology + branch lengths, sampled from the (possibly serially
  sampled) constant-population coalescent. ``ConstantCoalescent`` in
  ``treeflow.distributions.tree.coalescent.constant_coalescent`` only has a
  placeholder ``_sample_n`` ("Dummy sampling"), so ``simulate_coalescent_tree``
  below implements the standard forwards-in-time waiting-time algorithm
  instead. The random-pairing step (choose two lineages from the currently
  active set uniformly at random to coalesce) is the same idea used by
  ``build_random_topologies`` in the ``bd-tree-sampling`` branch's
  ``treeflow/distributions/tree/birthdeath/cpp_sampler.py``; that function
  assumes every leaf is active from time zero (contemporaneous sampling), so
  it doesn't directly support the serially-sampled case needed here and is
  reimplemented in NumPy below rather than adapted in place.
* an alignment, sampled by reusing ``treeflow.model.phylo_model.get_sequence_distribution``
  -- the exact same code path used to *score* likelihoods -- so simulated data
  is guaranteed consistent with the models the benchmarks evaluate.
"""
from __future__ import annotations

import types
import typing as tp

import dendropy
import numpy as np
import tensorflow as tf

from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.model.phylo_model import (
    PhyloModel,
    get_clock_model_rates,
    get_sequence_distribution,
    get_subst_model,
    get_subst_model_params,
    get_params,
)
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor
from treeflow.tree.io import tensor_to_dendro
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.tree.taxon_set import DictTaxonSet
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology

from benchmarks.params import get_return_value_of_empty_generator

_BASE_ORDER = "ACGT"
_ONE_HOT_TO_BASE = {tuple(row): base for row, base in zip(np.eye(4), _BASE_ORDER)}


def simulate_sampling_times(
    taxon_count: int, sampling_window: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly distributed serial sampling times over ``[0, sampling_window]``.

    A ``sampling_window`` of ``0`` gives contemporaneous (all-zero) sampling.
    """
    if sampling_window <= 0:
        return np.zeros(taxon_count)
    return rng.uniform(0.0, sampling_window, size=taxon_count)


def simulate_coalescent_tree(
    sampling_times: np.ndarray, pop_size: float, rng: np.random.Generator
) -> NumpyRootedTree:
    """Simulate a random topology and node heights under the constant-population
    (possibly heterochronous) coalescent, using the standard forwards-in-time
    waiting-time algorithm: draw an exponential coalescence waiting time at rate
    ``k * (k - 1) / (2 * pop_size)`` for the current number of active lineages
    ``k``, taking whichever of that event or the next sampling time comes first,
    and merging two uniformly-chosen active lineages at each coalescence.
    """
    taxon_count = sampling_times.shape[0]
    order = np.argsort(sampling_times, kind="stable")
    sorted_times = sampling_times[order]

    parent_indices = np.full(2 * taxon_count - 2, -1, dtype=np.int32)
    internal_heights = np.zeros(taxon_count - 1)

    active: tp.List[int] = []
    current_time = 0.0
    next_internal = taxon_count
    sample_ptr = 0

    while len(active) > 1 or sample_ptr < taxon_count:
        k = len(active)
        if k >= 2:
            rate = k * (k - 1) / (2.0 * pop_size)
            coalescent_time = current_time + rng.exponential(1.0 / rate)
        else:
            coalescent_time = np.inf

        if sample_ptr < taxon_count and sorted_times[sample_ptr] <= coalescent_time:
            current_time = sorted_times[sample_ptr]
            active.append(int(order[sample_ptr]))
            sample_ptr += 1
        else:
            current_time = coalescent_time
            i, j = rng.choice(k, size=2, replace=False)
            children = (active[i], active[j])
            for pos in sorted((i, j), reverse=True):
                active.pop(pos)
            new_node = next_internal
            internal_heights[new_node - taxon_count] = current_time
            parent_indices[children[0]] = new_node
            parent_indices[children[1]] = new_node
            active.append(new_node)
            next_internal += 1

    taxon_set = DictTaxonSet([f"taxon_{i}" for i in range(taxon_count)])
    topology = NumpyTreeTopology(parent_indices=parent_indices, taxon_set=taxon_set)
    return NumpyRootedTree(
        node_heights=internal_heights, sampling_times=sampling_times, topology=topology
    )


def simulate_alignment(
    tree: TensorflowRootedTree,
    phylo_model: PhyloModel,
    sequence_length: int,
    seed: int,
) -> np.ndarray:
    """Sample a one-hot encoded alignment (``[site, taxon, 4]``) along ``tree``
    under ``phylo_model``, by calling ``get_sequence_distribution`` -- the same
    function treeflow uses to build likelihoods -- and sampling from it.
    """
    subst_model = get_subst_model(phylo_model.subst_model)
    subst_model_params, _ = get_return_value_of_empty_generator(
        get_subst_model_params(phylo_model.subst_model, phylo_model.subst_params)
    )
    if phylo_model.site_model == "none":
        site_model_params: tp.Dict[str, object] = {}
    else:
        site_model_params, _ = get_return_value_of_empty_generator(
            get_params(phylo_model.site_params)
        )
    clock_model_params, _ = get_return_value_of_empty_generator(
        get_params(phylo_model.clock_params)
    )
    clock_model_rates = get_return_value_of_empty_generator(
        get_clock_model_rates(phylo_model.clock_model, clock_model_params, True, tree)
    )

    alignment_stub = types.SimpleNamespace(site_count=sequence_length)
    seq_dist = get_sequence_distribution(
        alignment_stub,
        tree,
        subst_model,
        subst_model_params,
        phylo_model.site_model,
        site_model_params,
        clock_model_rates,
    )
    return seq_dist.sample(seed=seed).numpy()


def write_fasta(encoded: np.ndarray, taxon_names: tp.Sequence[str], path: str) -> None:
    """Write a one-hot encoded alignment (``[site, taxon, 4]``) to FASTA."""
    site_count, taxon_count, _ = encoded.shape
    with open(path, "w") as f:
        for taxon_index, name in enumerate(taxon_names):
            bases = "".join(
                _ONE_HOT_TO_BASE[tuple(encoded[site, taxon_index].tolist())]
                for site in range(site_count)
            )
            f.write(f">{name}\n{bases}\n")


def write_newick(tree: NumpyRootedTree, taxon_names: tp.Sequence[str], path: str) -> None:
    """Write a simulated tree's topology and branch lengths to a newick file."""
    taxon_namespace = dendropy.TaxonNamespace(list(taxon_names))
    dendro_tree = tensor_to_dendro(
        tree.topology, taxon_namespace, list(taxon_names), tree.branch_lengths
    )
    dendro_tree.write(
        path=path, schema="newick", suppress_rooting=True, unquoted_underscores=True
    )


def simulate_height_samples(
    tree: TensorflowRootedTree, sample_count: int, height_scale: float, seed: int
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """Perturb ``tree``'s node heights in ratio space to build a batch of
    plausible tree states (e.g. as if drawn from an MCMC chain), giving
    likelihood/gradient benchmarks realistic branch-length variability rather
    than repeating one fixed input. Returns ``(branch_lengths, ratios)``, each
    with a leading batch dimension of size ``sample_count``.
    """
    anchor_heights = get_anchor_heights_tensor(tree.topology, tree.sampling_times)
    bij = NodeHeightRatioBijector(topology=tree.topology, anchor_heights=anchor_heights)
    loc = bij.inverse(tree.node_heights)
    base_dist = tf.random.stateless_normal(
        (sample_count,) + loc.shape, seed=[seed, 0], dtype=loc.dtype
    ) * tf.cast(height_scale, loc.dtype) + loc
    ratios = base_dist
    heights = bij.forward(ratios)
    sampled_trees = tree.with_node_heights(heights)
    return sampled_trees.branch_lengths, ratios


def simulate_replicate(
    taxon_count: int,
    pop_size: float,
    sampling_window: float,
    sim_model: PhyloModel,
    sequence_length: int,
    seed: int,
    tree_dir: str,
) -> tp.Tuple[str, str, TensorflowRootedTree]:
    """Simulate one tree + alignment, writing them to ``tree_dir`` as newick and
    FASTA files (so bito/BEAGLE-based benchmarkables, which need files on disk,
    can use them too). Returns ``(newick_file, fasta_file, tensor_tree)``.
    """
    import os

    rng = np.random.default_rng(seed)
    sampling_times = simulate_sampling_times(taxon_count, sampling_window, rng)
    numpy_tree = simulate_coalescent_tree(sampling_times, pop_size, rng)
    tensor_tree = convert_tree_to_tensor(numpy_tree)
    encoded = simulate_alignment(tensor_tree, sim_model, sequence_length, seed)

    taxon_names = [f"taxon_{i}" for i in range(taxon_count)]
    os.makedirs(tree_dir, exist_ok=True)
    newick_file = os.path.join(tree_dir, "tree.newick")
    fasta_file = os.path.join(tree_dir, "sequences.fasta")
    write_newick(numpy_tree, taxon_names, newick_file)
    write_fasta(encoded, taxon_names, fasta_file)
    return newick_file, fasta_file, tensor_tree
