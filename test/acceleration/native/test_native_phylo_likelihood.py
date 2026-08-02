"""Correctness tests for the native phylogenetic-likelihood custom op.

Everything is checked against the reference TensorFlow implementation in
``treeflow.traversal.phylo_likelihood`` (forward values and autodiff
gradients), plus the known HKY log-likelihood of the ``hello`` dataset.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.acceleration.native import native_phylogenetic_likelihood
from treeflow.traversal.phylo_likelihood import phylogenetic_likelihood as reference
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_eigen,
)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_matches_hky_beast(
    hello_tensor_tree, hello_alignment, hky_params, hello_hky_log_likelihood, function_mode
):
    """Native op reproduces the BEAST reference log-likelihood for `hello`."""
    subst_model = HKY()
    eigen = subst_model.eigen(**hky_params)
    probs = tf.expand_dims(
        get_transition_probabilities_eigen(eigen, hello_tensor_tree.branch_lengths), 0
    )
    encoded = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)

    func = tf.function(native_phylogenetic_likelihood) if function_mode else (
        native_phylogenetic_likelihood
    )
    site_partials = func(
        encoded,
        probs,
        hky_params["frequencies"],
        hello_tensor_tree.topology.postorder_node_indices,
        hello_tensor_tree.topology.node_child_indices,
        batch_shape=tf.shape(encoded)[:1],
    )
    res = tf.reduce_sum(tf.math.log(site_partials))
    assert_allclose(res.numpy(), hello_hky_log_likelihood)


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_matches_reference_forward(small_problem, function_mode):
    ref_fn = tf.function(reference) if function_mode else reference
    nat_fn = (
        tf.function(native_phylogenetic_likelihood)
        if function_mode
        else native_phylogenetic_likelihood
    )
    p = small_problem
    batch_shape = tf.shape(p["sequences"])[:1]
    ref_val = ref_fn(
        p["topology"], p["sequences"], p["transition_probs"], p["frequencies"],
        batch_shape=batch_shape,
    )
    nat_val = nat_fn(
        p["sequences"], p["transition_probs"], p["frequencies"],
        p["postorder_node_indices"], p["node_child_indices"], batch_shape=batch_shape,
    )
    assert_allclose(nat_val.numpy(), ref_val.numpy(), rtol=1e-12, atol=1e-12)


def test_native_float32(small_problem):
    p = small_problem
    seq = tf.cast(p["sequences"], tf.float32)
    probs = tf.cast(p["transition_probs"], tf.float32)
    freqs = tf.cast(p["frequencies"], tf.float32)
    nat = native_phylogenetic_likelihood(
        seq, probs, freqs, p["postorder_node_indices"], p["node_child_indices"]
    )
    ref = reference(
        p["topology"],
        seq,
        probs,
        freqs,
        batch_shape=tf.shape(seq)[:1],
    )
    assert nat.dtype == tf.float32
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient correctness (analytic native gradient vs reference autodiff)
# ---------------------------------------------------------------------------


def _grads(fn, problem, wrt_freqs=False, native=False):
    probs = tf.Variable(problem["transition_probs"])
    freqs = tf.Variable(problem["frequencies"])
    batch_shape = tf.shape(problem["sequences"])[:1]
    with tf.GradientTape() as tape:
        if native:
            site_partials = fn(
                problem["sequences"], probs, freqs,
                problem["postorder_node_indices"], problem["node_child_indices"],
                batch_shape=batch_shape,
            )
        else:
            site_partials = fn(
                problem["topology"], problem["sequences"], probs, freqs,
                batch_shape=batch_shape,
            )
        loss = tf.reduce_sum(tf.math.log(site_partials))
    targets = [probs, freqs] if wrt_freqs else [probs]
    return tape.gradient(loss, targets)


def test_native_gradient_matches_reference(small_problem):
    ref_grads = _grads(reference, small_problem, wrt_freqs=True, native=False)
    nat_grads = _grads(
        native_phylogenetic_likelihood, small_problem, wrt_freqs=True, native=True
    )
    for ref_g, nat_g in zip(ref_grads, nat_grads):
        assert nat_g is not None
        assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


def test_native_gradient_finite_difference(small_problem):
    """Spot-check the analytic gradient against central finite differences."""
    p = small_problem
    probs0 = p["transition_probs"]

    def loss_of(probs):
        site_partials = native_phylogenetic_likelihood(
            p["sequences"],
            probs,
            p["frequencies"],
            p["postorder_node_indices"],
            p["node_child_indices"],
        )
        return tf.reduce_sum(tf.math.log(site_partials))

    probs_var = tf.Variable(probs0)
    with tf.GradientTape() as tape:
        loss = loss_of(probs_var)
    analytic = tape.gradient(loss, probs_var).numpy()

    eps = 1e-6
    flat = probs0.numpy().reshape(-1).copy()
    rng = np.random.default_rng(0)
    idxs = rng.choice(flat.size, size=8, replace=False)
    for idx in idxs:
        plus = flat.copy()
        plus[idx] += eps
        minus = flat.copy()
        minus[idx] -= eps
        lp = loss_of(tf.constant(plus.reshape(probs0.shape))).numpy()
        lm = loss_of(tf.constant(minus.reshape(probs0.shape))).numpy()
        fd = (lp - lm) / (2 * eps)
        assert_allclose(
            analytic.reshape(-1)[idx], fd, rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_gradient_function_mode(small_problem, function_mode):
    fn = (
        tf.function(native_phylogenetic_likelihood)
        if function_mode
        else native_phylogenetic_likelihood
    )
    ref_grads = _grads(reference, small_problem, wrt_freqs=True, native=False)
    nat_grads = _grads(fn, small_problem, wrt_freqs=True, native=True)
    for ref_g, nat_g in zip(ref_grads, nat_grads):
        assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Site-blocking (SIMD) is purely a performance option: results must not change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [2, 8, 32])
def test_native_block_size_matches_unblocked(small_problem, block_size):
    p = small_problem
    args = (
        p["sequences"],
        p["transition_probs"],
        p["frequencies"],
        p["postorder_node_indices"],
        p["node_child_indices"],
    )
    base = native_phylogenetic_likelihood(*args, block_size=1)
    blocked = native_phylogenetic_likelihood(*args, block_size=block_size)
    assert_allclose(blocked.numpy(), base.numpy(), rtol=1e-15, atol=1e-18)

    def grads(block):
        probs = tf.Variable(p["transition_probs"])
        freqs = tf.Variable(p["frequencies"])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                tf.math.log(
                    native_phylogenetic_likelihood(
                        p["sequences"], probs, freqs,
                        p["postorder_node_indices"], p["node_child_indices"],
                        block_size=block,
                    )
                )
            )
        return tape.gradient(loss, [probs, freqs])

    for g0, gb in zip(grads(1), grads(block_size)):
        assert_allclose(gb.numpy(), g0.numpy(), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Batched transition probabilities (e.g. rate categories) -- Bt == B path
# ---------------------------------------------------------------------------


def test_native_batched_transition_probs(small_problem):
    """Per-batch (non-broadcast) transition matrices match the reference."""
    p = small_problem
    site_count = p["site_count"]
    node_count = 2 * p["leaf_count"] - 1
    state = p["state_count"]
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.1, 1.0, size=(site_count, node_count, state, state))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    probs = tf.constant(raw, dtype=p["sequences"].dtype)

    nat = native_phylogenetic_likelihood(
        p["sequences"],
        probs,
        p["frequencies"],
        p["postorder_node_indices"],
        p["node_child_indices"],
    )
    ref = reference(
        p["topology"],
        p["sequences"],
        probs,
        p["frequencies"],
        batch_shape=tf.shape(p["sequences"])[:1],
    )
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-11, atol=1e-11)

    # And its gradient.
    probs_var = tf.Variable(probs)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(
            tf.math.log(
                native_phylogenetic_likelihood(
                    p["sequences"],
                    probs_var,
                    p["frequencies"],
                    p["postorder_node_indices"],
                    p["node_child_indices"],
                )
            )
        )
    nat_g = tape.gradient(loss, probs_var)
    probs_ref = tf.Variable(probs)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(
            tf.math.log(
                reference(
                    p["topology"],
                    p["sequences"],
                    probs_ref,
                    p["frequencies"],
                    batch_shape=tf.shape(p["sequences"])[:1],
                )
            )
        )
    ref_g = tape.gradient(loss, probs_ref)
    assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Shared sub-batch transition probabilities (e.g. discrete rate categories):
# Bt = M < B, so the op gathers M matrix sets instead of tiling to sites * M.
# ---------------------------------------------------------------------------


def _rate_category_problem(small_problem, n_categories, seed=11):
    """Mimic a rate-category mixture: probs vary only across the categories.

    Returns broadcast-shaped tensors with batch ``[sites, M]`` where the
    transition matrices have batch ``[1, M]`` (one set per category, shared
    across sites) -- the pattern produced by ``DiscreteParameterMixture`` over
    ``LeafCTMC``.
    """
    p = small_problem
    sites = p["site_count"]
    node_count = 2 * p["leaf_count"] - 1
    state = p["state_count"]
    dtype = p["sequences"].dtype
    rng = np.random.default_rng(seed)

    raw = rng.uniform(0.1, 1.0, size=(1, n_categories, node_count, state, state))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    probs = tf.constant(raw, dtype=dtype)

    # Leaf data is identical across categories; full batch is [sites, M].
    seq = tf.broadcast_to(
        tf.expand_dims(p["sequences"], -3),
        tf.concat([[sites, n_categories], tf.shape(p["sequences"])[-2:]], axis=0),
    )
    full_batch = tf.constant([sites, n_categories])
    return p, seq, probs, p["frequencies"], full_batch


def test_native_shared_batch_matches_reference(small_problem):
    p, seq, probs, freqs, full_batch = _rate_category_problem(small_problem, 3)
    args = (seq, probs, freqs, p["postorder_node_indices"], p["node_child_indices"])

    nat = native_phylogenetic_likelihood(*args)
    ref = reference(p["topology"], seq, probs, freqs, batch_shape=full_batch)
    assert tuple(nat.shape) == tuple(full_batch.numpy())
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-11, atol=1e-11)


def test_native_shared_batch_gradient_matches_reference(small_problem):
    p, seq, probs, freqs, full_batch = _rate_category_problem(small_problem, 3)

    def grads(fn, use_batch_shape, native):
        probs_v = tf.Variable(probs)
        freqs_v = tf.Variable(freqs)
        with tf.GradientTape() as tape:
            kwargs = dict(batch_shape=full_batch) if use_batch_shape else {}
            if native:
                out = fn(
                    seq, probs_v, freqs_v,
                    p["postorder_node_indices"], p["node_child_indices"], **kwargs,
                )
            else:
                out = fn(p["topology"], seq, probs_v, freqs_v, **kwargs)
            loss = tf.reduce_sum(tf.math.log(out))
        return tape.gradient(loss, [probs_v, freqs_v])

    ref_g = grads(reference, use_batch_shape=True, native=False)
    nat_g = grads(native_phylogenetic_likelihood, use_batch_shape=False, native=True)
    for r, n in zip(ref_g, nat_g):
        assert n is not None
        assert n.shape == r.shape
        assert_allclose(n.numpy(), r.numpy(), rtol=1e-9, atol=1e-9)


def test_native_shared_batch_equals_tiled(small_problem):
    """Gathering Bt = M sets gives the same result as explicitly tiling to B."""
    p, seq, probs, freqs, full_batch = _rate_category_problem(small_problem, 4)
    sites, M = int(full_batch[0]), int(full_batch[1])

    shared = native_phylogenetic_likelihood(
        seq, probs, freqs, p["postorder_node_indices"], p["node_child_indices"]
    )
    # Explicitly tile to a distinct matrix set per (site, category) -> Bt == B.
    tiled_probs = tf.broadcast_to(
        probs, tf.concat([[sites, M], tf.shape(probs)[-3:]], axis=0)
    )
    tiled = native_phylogenetic_likelihood(
        seq, tiled_probs, freqs,
        p["postorder_node_indices"], p["node_child_indices"],
    )
    assert_allclose(shared.numpy(), tiled.numpy(), rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [2, 8])
def test_native_shared_batch_block_size_matches(small_problem, block_size):
    """Site-blocking is performance-only for the shared-batch path too."""
    p, seq, probs, freqs, _ = _rate_category_problem(small_problem, 3)
    idx = (p["postorder_node_indices"], p["node_child_indices"])

    base = native_phylogenetic_likelihood(seq, probs, freqs, *idx, block_size=1)
    blocked = native_phylogenetic_likelihood(
        seq, probs, freqs, *idx, block_size=block_size
    )
    assert_allclose(blocked.numpy(), base.numpy(), rtol=0, atol=0)

    def grad(block):
        probs_v = tf.Variable(probs)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(
                tf.math.log(
                    native_phylogenetic_likelihood(
                        seq, probs_v, freqs, *idx, block_size=block
                    )
                )
            )
        return tape.gradient(loss, probs_v)

    assert_allclose(grad(block_size).numpy(), grad(1).numpy(), rtol=1e-12, atol=1e-12)


def test_native_leading_batch_before_sites(small_problem):
    """Broadcast (site) axis in the *middle* of the batch, e.g. VI over a rate
    mixture: params drawn per-sample and per-category but shared across sites.

    Full batch is [n_samples, sites, M] while the transition matrices have batch
    [n_samples, 1, M] -- so the op must gather Bt = n_samples * M sets (not tile
    to n_samples * sites * M). A trailing-stride (b % Bt) shortcut would read the
    wrong matrices here; the dimension-aware gather index must get it right.
    """
    p = small_problem
    sites = p["site_count"]
    node_count = 2 * p["leaf_count"] - 1
    state = p["state_count"]
    leaf = p["leaf_count"]
    dtype = p["sequences"].dtype
    rng = np.random.default_rng(5)
    n_samples, M = 5, 3

    # Transition matrices vary per sample and per category, broadcast over sites.
    raw = rng.uniform(0.1, 1.0, size=(n_samples, 1, M, node_count, state, state))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    probs = tf.constant(raw, dtype=dtype)

    # Leaf data varies per site only; broadcast to the full [n_samples, sites, M].
    seq = tf.broadcast_to(
        p["sequences"][None, :, None],  # [1, sites, 1, leaf, state]
        [n_samples, sites, M, leaf, state],
    )
    freqs = p["frequencies"]
    full_batch = tf.constant([n_samples, sites, M])
    idx = (p["postorder_node_indices"], p["node_child_indices"])

    nat = native_phylogenetic_likelihood(seq, probs, freqs, *idx)
    ref = reference(p["topology"], seq, probs, freqs, batch_shape=full_batch)
    assert tuple(nat.shape) == (n_samples, sites, M)
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-11, atol=1e-11)

    def grad(fn, use_batch_shape, native):
        probs_v = tf.Variable(probs)
        with tf.GradientTape() as tape:
            kwargs = dict(batch_shape=full_batch) if use_batch_shape else {}
            if native:
                out = fn(seq, probs_v, freqs, *idx, **kwargs)
            else:
                out = fn(p["topology"], seq, probs_v, freqs, **kwargs)
            loss = tf.reduce_sum(tf.math.log(out))
        return tape.gradient(loss, probs_v)

    nat_g = grad(native_phylogenetic_likelihood, False, native=True)
    ref_g = grad(reference, True, native=False)
    # Gradient reduces back to the probs' own (sites-broadcast) shape.
    assert tuple(nat_g.shape) == tuple(probs.shape)
    assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


def test_native_shared_batch_rescaled_gradient_matches(small_problem):
    p, seq, probs, freqs, full_batch = _rate_category_problem(small_problem, 3)

    def grads(fn, log_output, use_batch_shape):
        probs_v = tf.Variable(probs)
        freqs_v = tf.Variable(freqs)
        with tf.GradientTape() as tape:
            kwargs = dict(batch_shape=full_batch) if use_batch_shape else {}
            out = fn(
                seq, probs_v, freqs_v,
                p["postorder_node_indices"], p["node_child_indices"], **kwargs,
            )
            ll = out if log_output else tf.math.log(out)
            loss = tf.reduce_sum(ll)
        return tape.gradient(loss, [probs_v, freqs_v])

    ref_g = grads(native_phylogenetic_likelihood, log_output=False, use_batch_shape=False)
    nat_g = grads(
        native_phylogenetic_log_likelihood_rescaled, log_output=True, use_batch_shape=False
    )
    for r, n in zip(ref_g, nat_g):
        assert_allclose(n.numpy(), r.numpy(), rtol=1e-8, atol=1e-9)


# ---------------------------------------------------------------------------
# Integration through the LeafCTMC distribution
# ---------------------------------------------------------------------------


def test_leaf_ctmc_rate_category_mixture_native_matches_reference(
    hello_tensor_tree, hello_alignment, hky_params
):
    """Native vs reference for a discrete-Gamma rate-category mixture.

    Exercises the full path where ``DiscreteParameterMixture`` introduces a
    category batch dimension, so the transition matrices are shared across
    sites (gathered as Bt = M sets) rather than tiled to sites * M.
    """
    from tensorflow_probability.python.distributions import Sample
    from treeflow.distributions.leaf_ctmc import LeafCTMC
    from treeflow.distributions.discrete_parameter_mixture import (
        DiscreteParameterMixture,
    )
    from treeflow.evolution.substitution.probabilities import (
        get_transition_probabilities_tree,
    )
    from treeflow.model.phylo_model import (
        get_discrete_gamma_site_rate_distribution,
    )

    dtype = hky_params["frequencies"].dtype
    site_rates = get_discrete_gamma_site_rate_distribution(
        category_count=4, site_gamma_shape=tf.constant(0.7, dtype=dtype)
    )

    def make_dist(use_native):
        transition_probs_tree = get_transition_probabilities_tree(
            hello_tensor_tree.get_unrooted_tree(),
            HKY(),
            rate_categories=site_rates.normalised_support,
            **hky_params,
        )
        return Sample(
            DiscreteParameterMixture(
                site_rates,
                LeafCTMC(
                    transition_probs_tree,
                    tf.expand_dims(hky_params["frequencies"], -2),
                    use_native=use_native,
                ),
            ),
            sample_shape=hello_alignment.site_count,
        )

    sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    native_val = make_dist(use_native=True).log_prob(sequences)
    ref_val = make_dist(use_native=False).log_prob(sequences)
    assert_allclose(native_val.numpy(), ref_val.numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("function_mode", [False, True])
def test_leaf_ctmc_use_native(
    hello_tensor_tree, hello_alignment, hky_params, hello_hky_log_likelihood, function_mode
):
    from tensorflow_probability.python.distributions import Sample
    from treeflow.distributions.leaf_ctmc import LeafCTMC
    from treeflow.evolution.substitution.probabilities import (
        get_transition_probabilities_tree,
    )

    def log_prob_fn(tree, sequences):
        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(), HKY(), **hky_params
        )
        dist = Sample(
            LeafCTMC(transition_prob_tree, hky_params["frequencies"], use_native=True),
            sample_shape=hello_alignment.site_count,
        )
        return dist.log_prob(sequences)

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)
    if function_mode:
        log_prob_fn = tf.function(log_prob_fn)
    res = log_prob_fn(hello_tensor_tree, sequences)
    assert_allclose(res.numpy(), hello_hky_log_likelihood)


# ---------------------------------------------------------------------------
# Rescaled (numerically stable) native op
# ---------------------------------------------------------------------------


from treeflow.acceleration.native import (  # noqa: E402
    native_phylogenetic_log_likelihood_rescaled,
)


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_rescaled_matches_unrescaled_log(small_problem, function_mode):
    p = small_problem
    fn = (
        tf.function(native_phylogenetic_log_likelihood_rescaled)
        if function_mode
        else native_phylogenetic_log_likelihood_rescaled
    )
    rescaled_log = fn(
        p["sequences"], p["transition_probs"], p["frequencies"],
        p["postorder_node_indices"], p["node_child_indices"],
    )
    unrescaled = native_phylogenetic_likelihood(
        p["sequences"], p["transition_probs"], p["frequencies"],
        p["postorder_node_indices"], p["node_child_indices"],
    )
    assert_allclose(rescaled_log.numpy(), tf.math.log(unrescaled).numpy(),
                    rtol=1e-11, atol=1e-11)


def test_native_rescaled_gradient_matches(small_problem):
    p = small_problem

    def log_loss(fn, log_output):
        probs = tf.Variable(p["transition_probs"])
        freqs = tf.Variable(p["frequencies"])
        with tf.GradientTape() as tape:
            out = fn(p["sequences"], probs, freqs,
                     p["postorder_node_indices"], p["node_child_indices"])
            ll = out if log_output else tf.math.log(out)
            loss = tf.reduce_sum(ll)
        return tape.gradient(loss, [probs, freqs])

    ref_g = log_loss(native_phylogenetic_likelihood, log_output=False)
    nat_g = log_loss(native_phylogenetic_log_likelihood_rescaled, log_output=True)
    for r, n in zip(ref_g, nat_g):
        assert_allclose(n.numpy(), r.numpy(), rtol=1e-8, atol=1e-9)


def test_native_rescaled_avoids_underflow(make_large_problem):
    """On a large tree the unrescaled likelihood underflows but rescaled does not."""
    p = make_large_problem(leaf_count=600, state_count=4, site_count=4)
    unrescaled = native_phylogenetic_likelihood(
        p["sequences"], p["transition_probs"], p["frequencies"],
        p["postorder_node_indices"], p["node_child_indices"],
    )
    rescaled_log = native_phylogenetic_log_likelihood_rescaled(
        p["sequences"], p["transition_probs"], p["frequencies"],
        p["postorder_node_indices"], p["node_child_indices"],
    )
    # Unrescaled underflows to zero (log -> -inf); rescaled stays finite.
    assert np.all(unrescaled.numpy() == 0.0)
    assert np.all(np.isfinite(rescaled_log.numpy()))


@pytest.mark.parametrize("use_native", [False, True])
@pytest.mark.parametrize("rescaling", [True, "auto", "adaptive"])
def test_leaf_ctmc_rescaling(
    hello_tensor_tree, hello_alignment, hky_params, hello_hky_log_likelihood,
    use_native, rescaling,
):
    from tensorflow_probability.python.distributions import Sample
    from treeflow.distributions.leaf_ctmc import LeafCTMC
    from treeflow.evolution.substitution.probabilities import (
        get_transition_probabilities_tree,
    )

    transition_prob_tree = get_transition_probabilities_tree(
        hello_tensor_tree.get_unrooted_tree(), HKY(), **hky_params
    )
    dist = Sample(
        LeafCTMC(transition_prob_tree, hky_params["frequencies"],
                 use_native=use_native, rescaling=rescaling),
        sample_shape=hello_alignment.site_count,
    )
    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)
    res = dist.log_prob(sequences)
    assert_allclose(res.numpy(), hello_hky_log_likelihood)


# ---------------------------------------------------------------------------
# Auto-detection of the native acceleration in LeafCTMC
# ---------------------------------------------------------------------------


def test_native_acceleration_available_true_when_built():
    from treeflow.distributions.leaf_ctmc import native_acceleration_available

    # This module only runs when the native op is built (see conftest), so
    # auto-detection must report it as available.
    assert native_acceleration_available() is True


def test_leaf_ctmc_auto_uses_native_when_available(
    hello_tensor_tree, hky_params
):
    from treeflow.distributions.leaf_ctmc import LeafCTMC
    from treeflow.evolution.substitution.probabilities import (
        get_transition_probabilities_tree,
    )

    transition_prob_tree = get_transition_probabilities_tree(
        hello_tensor_tree.get_unrooted_tree(), HKY(), **hky_params
    )
    # Defaults: use_native="auto", rescaling="adaptive".
    dist = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
    assert dist._use_native is True
    assert dist.rescaling == "adaptive"

    forced_off = LeafCTMC(
        transition_prob_tree, hky_params["frequencies"], use_native=False
    )
    assert forced_off._use_native is False


def test_leaf_ctmc_default_matches_reference(
    hello_tensor_tree, hello_alignment, hky_params, hello_hky_log_likelihood
):
    """The auto/adaptive defaults give the correct log-likelihood."""
    from tensorflow_probability.python.distributions import Sample
    from treeflow.distributions.leaf_ctmc import LeafCTMC
    from treeflow.evolution.substitution.probabilities import (
        get_transition_probabilities_tree,
    )

    transition_prob_tree = get_transition_probabilities_tree(
        hello_tensor_tree.get_unrooted_tree(), HKY(), **hky_params
    )
    dist = Sample(
        LeafCTMC(transition_prob_tree, hky_params["frequencies"]),
        sample_shape=hello_alignment.site_count,
    )
    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)
    assert_allclose(dist.log_prob(sequences).numpy(), hello_hky_log_likelihood)
