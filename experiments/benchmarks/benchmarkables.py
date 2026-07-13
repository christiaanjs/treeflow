"""Benchmarkable implementations: treeflow (TF graph, with an optional native
C++-op fast path), JAX (via phylojax, if installed), and BEAGLE/bito (if
installed).

Ported and simplified from ``treeflow_benchmarks/{treeflow,jax,bito,bito_direct}.py``.
The treeflow implementation now builds its log-probability with
``treeflow.model.phylo_model.get_sequence_distribution`` (the same function
used to build production likelihoods) instead of duplicating that logic, and
takes a ``use_native`` flag so the identical benchmarkable can run with or
without the native C++ ops (``treeflow.acceleration.native``) -- both
``LeafCTMC`` and ``NodeHeightRatioBijector`` already support a ``use_native``
switch, so no new native-op plumbing is needed here.
"""
import typing as tp

import numpy as np
import tensorflow as tf

from benchmarks import benchmarking as bench
from benchmarks.params import (
    get_return_value_of_empty_generator,
    merge_params,
    split_gradient_params,
)
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.evolution.seqio import Alignment
from treeflow.model.phylo_model import (
    PhyloModel,
    get_clock_model_rates,
    get_sequence_distribution,
    get_subst_model,
)
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor


class TreeflowLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    """TensorFlow-graph likelihood/gradient, optionally routed through the
    native C++ pruning op (``use_native=True``)."""

    def __init__(self, use_native: tp.Union[str, bool] = False):
        self.use_native = use_native

    def initialize(self, newick_file, fasta_file, model, calculate_clock_rate_gradient):
        self.tree = convert_tree_to_tensor(parse_newick(newick_file))
        unrooted_tree = self.tree.get_unrooted_tree()
        alignment = Alignment(fasta_file).get_compressed_alignment()
        sequences_encoded = alignment.get_encoded_sequence_tensor(self.tree.taxon_set)
        pattern_counts = alignment.get_weights_tensor()

        phylo_model = PhyloModel(model)
        subst_model = get_subst_model(phylo_model.subst_model)
        self.gradient_params, self.non_gradient_params = split_gradient_params(
            phylo_model, calculate_clock_rate_gradient
        )
        use_native = self.use_native

        def log_prob(branch_lengths, gradient_params):
            tree = unrooted_tree.with_branch_lengths(branch_lengths)
            params = merge_params(gradient_params, self.non_gradient_params)
            clock_model_rates = get_return_value_of_empty_generator(
                get_clock_model_rates(
                    phylo_model.clock_model, params["clock_model_params"], True, self.tree
                )
            )
            seq_dist = get_sequence_distribution(
                alignment,
                tree,
                subst_model,
                params["subst_model_params"],
                phylo_model.site_model,
                params["site_model_params"],
                clock_model_rates,
                pattern_counts=pattern_counts,
                use_native=use_native,
            )
            return seq_dist.log_prob(sequences_encoded)

        self.log_prob = tf.function(log_prob)

        def grad(branch_lengths, params):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                tf.nest.map_structure(t.watch, params)
                log_prob_val = self.log_prob(branch_lengths, params)
            return t.gradient(log_prob_val, [branch_lengths, params])

        self.grad = tf.function(grad)

        branch_lengths = self.tree.branch_lengths
        self.log_prob(branch_lengths, self.gradient_params)  # trace/compile
        self.grad(branch_lengths, self.gradient_params)

    def calculate_likelihoods(self, branch_lengths: np.ndarray, params: object) -> np.ndarray:
        gradient_params_tensor = tf.nest.map_structure(
            lambda x, y: tf.constant(x, dtype=y.dtype), params, self.gradient_params
        )
        branch_lengths_tensor = tf.constant(branch_lengths, dtype=self.tree.heights.dtype)
        return self.log_prob(branch_lengths_tensor, gradient_params_tensor).numpy()

    def calculate_gradients(self, branch_lengths: np.ndarray, params: object):
        gradient_params_tensor = tf.nest.map_structure(
            lambda x, y: tf.constant(x, dtype=y.dtype), params, self.gradient_params
        )
        branch_lengths_tensor = tf.constant(branch_lengths, dtype=self.tree.heights.dtype)
        tensor_grad = self.grad(branch_lengths_tensor, gradient_params_tensor)
        return tf.nest.map_structure(lambda x: x.numpy(), tensor_grad)


class TreeflowRatioTransformBenchmarkable(bench.RatioTransformBenchmarkable):
    """Node-height ratio transform forward/gradient, optionally routed through
    the native C++ op (``use_native=True``)."""

    def __init__(self, use_native: tp.Union[str, bool] = False):
        self.use_native = use_native

    def initialize(self, newick_file):
        tree = convert_tree_to_tensor(parse_newick(newick_file))
        anchor_heights = get_anchor_heights_tensor(tree.topology, tree.sampling_times)
        self.bij = NodeHeightRatioBijector(
            topology=tree.topology, anchor_heights=anchor_heights, use_native=self.use_native
        )
        self.forward = tf.function(self.bij.forward)

        def grad(ratios, height_gradients):
            with tf.GradientTape() as t:
                t.watch(ratios)
                heights = self.bij.forward(ratios)
            return t.gradient(heights, ratios, output_gradients=height_gradients)

        self.grad = tf.function(grad)

        ratios = self.bij.inverse(tree.node_heights)
        self.ratios = ratios
        self.forward(ratios)  # trace/compile
        self.grad(ratios, tf.ones_like(ratios))

    def calculate_heights(self, ratios: np.ndarray):
        return self.forward(tf.constant(ratios, dtype=self.ratios.dtype))

    def calculate_ratio_gradients(self, ratios: np.ndarray, height_gradients: np.ndarray):
        return self.grad(
            tf.constant(ratios, dtype=self.ratios.dtype),
            tf.constant(height_gradients, dtype=self.ratios.dtype),
        )


class JaxLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    """JAX likelihood/gradient via phylojax (https://github.com/christiaanjs/phylojax)."""

    def __init__(self, jit: bool = False):
        self.jit = jit

    def initialize(self, newick_file, fasta_file, model, calculate_clock_rate_gradient):
        import jax
        import jax.numpy as jnp
        import phylojax.likelihood
        import phylojax.site_rate_variation
        import phylojax.substitution

        from benchmarks.params import get_numpy_gradient_params_dict

        subst_model_classes = dict(
            hky=phylojax.substitution.HKY,
            jc=phylojax.substitution.JC,
            gtr=phylojax.substitution.GTR,
        )

        phylo_model = PhyloModel(model)
        params = get_numpy_gradient_params_dict(phylo_model, calculate_clock_rate_gradient)
        tree = parse_newick(newick_file)
        compressed_alignment = Alignment(fasta_file).get_compressed_alignment()
        encoded_sequences = compressed_alignment.get_encoded_sequence_array(tree.taxon_set)
        encoded_sequences_node_first = jnp.moveaxis(encoded_sequences, 1, 0)
        pattern_counts = compressed_alignment.get_weights_array()
        default_weights = jnp.array([1.0])
        default_rates = jnp.array([1.0])
        topology_dict = dict(
            child_indices=tree.topology.child_indices,
            postorder_node_indices=tree.topology.postorder_node_indices,
        )

        def log_prob(branch_lengths, params):
            if phylo_model.site_model == "discrete_gamma":
                category_weights, category_rates = (
                    phylojax.site_rate_variation.get_discrete_gamma_weights_rates(
                        **params["site_model_params"],
                        category_count=phylo_model.site_params["category_count"],
                    )
                )
            elif phylo_model.site_model == "discrete_weibull":
                category_weights, category_rates = (
                    phylojax.site_rate_variation.get_discrete_weibull_weights_rates(
                        **params["site_model_params"],
                        category_count=phylo_model.site_params["category_count"],
                    )
                )
            elif phylo_model.site_model == "none":
                category_weights, category_rates = default_weights, default_rates
            else:
                raise ValueError(f"Unknown site model: {phylo_model.site_model}")

            subst_model = subst_model_classes[phylo_model.subst_model](
                **params["subst_model_params"]
            )
            if phylo_model.clock_model != "strict":
                raise ValueError(f"Unknown clock model: {phylo_model.clock_model}")
            if calculate_clock_rate_gradient:
                rates = params["clock_model_params"]["clock_rate"]
            else:
                rates = phylo_model.clock_params["clock_rate"]

            likelihood = phylojax.likelihood.JaxLikelihood(
                topology_dict,
                encoded_sequences_node_first,
                subst_model,
                pattern_counts,
                category_weights,
                category_rates,
            )
            return likelihood.log_likelihood(branch_lengths * rates)

        grad = lambda *args: list(jax.grad(log_prob, argnums=[0, 1])(*args))

        if self.jit:
            self.log_prob = jax.jit(log_prob)
            self.grad = jax.jit(grad)
        else:
            self.log_prob = log_prob
            self.grad = grad

        # JAX dispatches asynchronously: a (jitted) call returns a future
        # immediately while the computation runs in the background. Without
        # forcing completion we would time only Python dispatch latency
        # (~0.2 ms) rather than the real compute (tens of ms), making JAX --
        # especially jax_jit -- look dramatically, and spuriously, faster than
        # the native op (which is forced to complete by ``.numpy()``). Block on
        # the results so JAX is timed on the same footing.
        self._block = jax.block_until_ready

        self.params = params
        branch_lengths = tree.branch_lengths
        self.log_prob(branch_lengths, params)  # trace/compile
        self.grad(branch_lengths, params)

    def calculate_likelihoods(self, branch_lengths, params):
        return self._block(self.log_prob(branch_lengths, params))

    def calculate_gradients(self, branch_lengths, params):
        return self._block(self.grad(branch_lengths, params))


def bito_available() -> bool:
    try:
        import bito  # noqa: F401

        return True
    except ImportError:
        return False


def build_bito_benchmarkables():
    """Only call once ``bito_available()`` is True."""
    import os

    from treeflow.acceleration.bito.beagle import phylogenetic_likelihood
    from treeflow.acceleration.bito.instance import get_instance
    from treeflow.acceleration.bito.ratio_transform import ratios_to_node_heights
    from treeflow.model.phylo_model import get_subst_model_params

    from benchmarks.simulate import DATES_CSV_FILENAME

    def _dates_csv_for(newick_file):
        """The ``dates.csv`` ``simulate_replicate`` wrote alongside
        ``newick_file`` (see ``benchmarks.simulate.write_dates_csv``). Simulated
        taxa are named ``taxon_{i}``, whose trailing index bito's own
        ``parse_dates_from_taxon_names`` would misread as a date, so bito-based
        benchmarkables read tip dates from this CSV instead."""
        return os.path.join(os.path.dirname(newick_file), DATES_CSV_FILENAME)

    class BeagleLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
        """BEAGLE/bito likelihood driven through the TreeFlow ``tf.function``
        wrapper (``treeflow.acceleration.bito.beagle.phylogenetic_likelihood``),
        so timings include TensorFlow's per-call graph-dispatch and tensor <->
        numpy marshalling overhead -- the same footing as the ``treeflow``
        methods."""

        def initialize(self, newick_file, fasta_file, model, calculate_clock_rate_gradient):
            self.phylo_model = PhyloModel(model)
            self.calculate_clock_rate_gradient = calculate_clock_rate_gradient
            subst_model = get_subst_model(self.phylo_model.subst_model)
            subst_params, _ = get_return_value_of_empty_generator(
                get_subst_model_params(
                    self.phylo_model.subst_model, self.phylo_model.subst_params
                )
            )
            inst = get_instance(newick_file, dates_csv=_dates_csv_for(newick_file))
            log_prob, self.inst = phylogenetic_likelihood(
                fasta_file,
                subst_model,
                inst=inst,
                clock_rate=self.phylo_model.clock_params["clock_rate"],
                site_model=self.phylo_model.site_model,
                site_model_params=self.phylo_model.site_params,
                **subst_params,
            )
            self.log_prob = tf.function(lambda branch_lengths: log_prob(branch_lengths))

            def grad(branch_lengths):
                with tf.GradientTape() as t:
                    t.watch(branch_lengths)
                    log_prob_val = self.log_prob(branch_lengths)
                return t.gradient(log_prob_val, branch_lengths)

            self.grad = tf.function(grad)
            branch_lengths = np.array(
                self.inst.tree_collection.trees[0].branch_lengths
            )[:-1].copy()
            self.log_prob(branch_lengths)
            self.grad(branch_lengths)

        def calculate_likelihoods(self, branch_lengths, params):
            return self.log_prob(branch_lengths).numpy()

        def calculate_gradients(self, branch_lengths, params):
            return self.grad(branch_lengths).numpy()

    class BeagleDirectLikelihoodBenchmarkable(BeagleLikelihoodBenchmarkable):
        """BEAGLE/bito likelihood driven *directly* on the bito instance,
        bypassing the TreeFlow ``tf.function`` wrapper. Branch lengths are
        written into bito's own (in-place) state array and
        ``inst.log_likelihoods()`` / ``inst.phylo_gradients()`` are called
        straight through, so this isolates BEAGLE's compute from the TensorFlow
        graph-dispatch and tensor<->numpy marshalling overhead that
        ``beagle_bito`` incurs -- a lower bound on what bito can deliver.
        Ported from the old ``treeflow_benchmarks/bito_direct.py``."""

        def initialize(self, newick_file, fasta_file, model, calculate_clock_rate_gradient):
            super().initialize(newick_file, fasta_file, model, calculate_clock_rate_gradient)
            # A mutable numpy view into bito's internal branch-length state; the
            # final entry is the root's (unused) branch, so only [:-1] is set.
            self.branch_length_state = np.array(
                self.inst.tree_collection.trees[0].branch_lengths, copy=False
            )

        def calculate_likelihoods(self, branch_lengths, params):
            self.branch_length_state[:-1] = branch_lengths
            return np.array(self.inst.log_likelihoods())[0]

        def _extract_substitution_model_grads(self, bito_gradient, subst_grad_dict):
            subst_model = self.phylo_model.subst_model
            if subst_model == "gtr":
                subst_grad_dict["rates"] = np.array(
                    bito_gradient["substitution_model"]
                )[:-2]
            elif subst_model == "hky":
                subst_grad_dict["kappa"] = np.array(
                    bito_gradient["substitution_model"]
                )[0]
            if subst_model != "jc":
                subst_grad_dict["frequencies"] = np.array(
                    bito_gradient["substitution_model"]
                )[-4:]

        def calculate_gradients(self, branch_lengths, params):
            self.branch_length_state[:-1] = branch_lengths
            gradient = self.inst.phylo_gradients()[0]
            branch_gradient_array = np.array(gradient.gradient["branch_lengths"])
            param_gradient = dict(
                clock_model_params=dict(),
                subst_model_params=dict(),
                site_model_params=dict(),
            )
            if (
                self.calculate_clock_rate_gradient
                and "clock_rate" in params["clock_model_params"]
            ):
                param_gradient["clock_model_params"]["clock_rate"] = np.array(
                    gradient.gradient["clock_model"]
                )
            if "site_weibull_concentration" in params["site_model_params"]:
                param_gradient["site_model_params"][
                    "site_weibull_concentration"
                ] = np.array(gradient.gradient["site_model"])
            self._extract_substitution_model_grads(
                gradient.gradient, param_gradient["subst_model_params"]
            )
            return [branch_gradient_array[:-1], param_gradient]

    class BitoRatioTransformBenchmarkable(bench.RatioTransformBenchmarkable):
        def initialize(self, newick_file):
            self.inst = get_instance(newick_file, dates_csv=_dates_csv_for(newick_file))
            self.tree = self.inst.tree_collection.trees[0]
            self.anchor_heights = np.array(self.tree.node_heights, copy=False)

        def calculate_heights(self, ratios):
            return ratios_to_node_heights(self.inst, self.anchor_heights, ratios)

        def calculate_ratio_gradients(self, ratios, height_gradients):
            with tf.GradientTape() as t:
                ratios_t = tf.constant(ratios)
                t.watch(ratios_t)
                heights = ratios_to_node_heights(self.inst, self.anchor_heights, ratios_t)
            return t.gradient(heights, ratios_t, output_gradients=height_gradients)

    return dict(
        likelihood=BeagleLikelihoodBenchmarkable(),
        likelihood_direct=BeagleDirectLikelihoodBenchmarkable(),
        ratio_transform=BitoRatioTransformBenchmarkable(),
    )


def phylojax_available() -> bool:
    try:
        import phylojax  # noqa: F401

        return True
    except ImportError:
        return False


def build_likelihood_benchmarkables() -> tp.Dict[str, bench.LikelihoodBenchmarkable]:
    benchmarkables = dict(
        treeflow=TreeflowLikelihoodBenchmarkable(use_native=False),
        treeflow_native=TreeflowLikelihoodBenchmarkable(use_native=True),
    )
    if phylojax_available():
        benchmarkables["jax"] = JaxLikelihoodBenchmarkable(jit=False)
        benchmarkables["jax_jit"] = JaxLikelihoodBenchmarkable(jit=True)
    if bito_available():
        bito_benchmarkables = build_bito_benchmarkables()
        benchmarkables["beagle_bito"] = bito_benchmarkables["likelihood"]
        benchmarkables["beagle_bito_direct"] = bito_benchmarkables["likelihood_direct"]
    return benchmarkables


def build_ratio_transform_benchmarkables() -> tp.Dict[str, bench.RatioTransformBenchmarkable]:
    benchmarkables = dict(
        treeflow=TreeflowRatioTransformBenchmarkable(use_native=False),
        treeflow_native=TreeflowRatioTransformBenchmarkable(use_native=True),
    )
    if bito_available():
        benchmarkables["beagle_bito"] = build_bito_benchmarkables()["ratio_transform"]
    return benchmarkables
