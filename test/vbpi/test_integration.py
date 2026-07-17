"""End-to-end VBPI loop: SBN + branch model + VIMCO on a toy target.

The target is a *known, normalised* joint over (topology, branch lengths):
a fixed "true" SBN over topologies times fixed per-split log-normals over branch
lengths. The variational family is exactly of this form, so the true posterior
is in the family and the maximal K-sample bound is ``log Z = 0``. A few VIMCO
steps must increase the bound and drive the topology parameters towards the
truth, exercising the whole pipeline (native/NumPy sampling, differentiable
``log_prob``, reparameterised branches, VIMCO gradients for both parameter sets).
"""
import numpy as np
import tensorflow as tf

from treeflow.vbpi.branch_model import SplitLognormalBranchModel
from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices
from treeflow.vbpi.vimco import vimco_surrogate


def _build(taxon_count, seed):
    topos = all_rooted_parent_indices(taxon_count)
    support = SubsplitSupport.from_topologies(topos, taxon_count)
    rng = np.random.default_rng(seed)
    # "True" generative parameters.
    true_logits = tf.constant(rng.normal(size=support.num_candidates))
    true_loc = tf.constant(rng.normal(size=support.num_clades) * 0.3 - 1.5)
    true_scale = tf.constant(0.2 + 0.1 * rng.random(support.num_clades))
    return support, true_logits, true_loc, true_scale


def _target_log_prob(support, true_logits, true_loc, true_scale,
                     candidate_indices, node_clade_ids, branch_lengths):
    """log p(T, b) under the fixed, normalised true joint."""
    true_sbn = SubsplitBayesianNetwork(
        support, logits=tf.Variable(true_logits)
    )
    log_p_topology = true_sbn.log_prob(candidate_indices)
    branch_clade_ids = node_clade_ids[..., : 2 * support.taxon_count - 2]
    import tensorflow_probability as tfp

    dist = tfp.distributions.Independent(
        tfp.distributions.LogNormal(
            loc=tf.gather(true_loc, branch_clade_ids),
            scale=tf.gather(true_scale, branch_clade_ids),
        ),
        reinterpreted_batch_ndims=1,
    )
    return log_p_topology + dist.log_prob(branch_lengths)


def test_vbpi_elbo_increases():
    tf.random.set_seed(0)
    taxon_count = 5
    support, true_logits, true_loc, true_scale = _build(taxon_count, seed=1)

    sbn = SubsplitBayesianNetwork(support)  # zero logits (uniform-ish start)
    branch_model = SplitLognormalBranchModel(support)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    variables = [sbn.logits, branch_model.loc, branch_model.scale_param]

    K = 8

    def step(seed):
        samples = sbn.sample_topologies(K, seed=seed, use_native=False)
        candidate_indices = tf.constant(samples.candidate_indices.astype(np.int64))
        node_clade_ids = tf.constant(samples.node_clade_ids)
        with tf.GradientTape() as tape:
            log_q_topology = sbn.log_prob(candidate_indices)  # [K]
            branch_lengths, log_q_branch = branch_model.sample_and_log_prob(
                node_clade_ids, seed=(seed, 17)
            )
            log_p = _target_log_prob(
                support, true_logits, true_loc, true_scale,
                candidate_indices, node_clade_ids, branch_lengths,
            )
            log_weights = log_p - log_q_topology - log_q_branch  # [K], attached
            obj = vimco_surrogate(
                log_weights[tf.newaxis], log_q_topology[tf.newaxis]
            )
            loss = -tf.reduce_mean(obj.surrogate)
        grads = tape.gradient(loss, variables)
        assert all(g is not None for g in grads)
        # The SBN logit gradient arrives as IndexedSlices (via tf.gather); densify
        # before checking finiteness.
        assert all(
            np.all(np.isfinite(tf.convert_to_tensor(g).numpy())) for g in grads
        )
        optimizer.apply_gradients(zip(grads, variables))
        return float(tf.reduce_mean(obj.elbo))

    early = np.mean([step(seed) for seed in range(1, 21)])
    for seed in range(21, 200):
        step(seed)
    late = np.mean([step(seed) for seed in range(200, 220)])

    # The bound is <= log Z = 0 and should have increased substantially.
    assert late > early
    assert late < 0.05  # never exceeds the true log marginal (0) beyond MC noise
