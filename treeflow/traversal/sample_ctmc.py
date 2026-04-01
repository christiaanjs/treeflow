import tensorflow as tf
from tensorflow_probability.python.internal import samplers


@tf.function
def sample_ctmc_preorder(
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    preorder_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    taxon_count: tf.Tensor,
    n: tf.Tensor,
    batch_shape: tf.Tensor,
    seed=None,
) -> tf.Tensor:
    """
    Sample leaf sequences from a phylogenetic CTMC via preorder traversal.

    Parameters
    ----------
    transition_probs
        Shape [...batch, n_branches, state, state].
        transition_probs[..., i, s, t] = P(child state = t | parent state = s) on branch i.
        Branch index i corresponds to node i (for all non-root nodes).
    frequencies
        Shape [...batch, state]. Equilibrium frequencies used to sample the root state.
    preorder_indices
        Shape [2n-1]. All node indices in preorder order (root first).
    parent_indices
        Shape [2n-2]. parent_indices[i] is the parent node index of node i.
    taxon_count
        Number of leaf taxa (n).
    n
        Number of independent samples to draw.
    batch_shape
        Batch shape as a 1-D tensor.
    seed
        Optional seed for stateless random ops.

    Returns
    -------
    tf.Tensor
        Shape [n, ...batch, taxon_count, state_count]. One-hot encoded leaf sequences
        as tf.int32.
    """
    node_count = 2 * taxon_count - 1
    state_count = tf.shape(frequencies)[-1]
    sample_and_batch_shape = tf.concat([[n], batch_shape], axis=0)
    flat_size = tf.reduce_prod(sample_and_batch_shape)

    # Split seed: one per node (root + 2n-2 non-root nodes = 2n-1 total)
    all_seeds = samplers.split_seed(seed, n=node_count)

    # ------------------------------------------------------------------ #
    # Root sampling                                                        #
    # ------------------------------------------------------------------ #
    # Broadcast frequencies to [n, ...batch, state] then flatten to [N, state]
    freq_b = tf.broadcast_to(
        frequencies,
        tf.concat([sample_and_batch_shape, [state_count]], axis=0),
    )
    freq_flat = tf.reshape(freq_b, [flat_size, state_count])  # [N, state]
    root_state_flat = tf.cast(
        tf.random.stateless_categorical(tf.math.log(freq_flat), 1, seed=all_seeds[0])[
            :, 0
        ],
        tf.int32,
    )  # [N]
    root_state = tf.reshape(root_state_flat, sample_and_batch_shape)  # [n, ...batch]

    # ------------------------------------------------------------------ #
    # TensorArray: one entry per node, each holding [n, ...batch] states  #
    # ------------------------------------------------------------------ #
    states_ta = tf.TensorArray(
        dtype=tf.int32,
        size=node_count,
        clear_after_read=False,
        element_shape=None,
    )
    root_index = node_count - 1
    states_ta = states_ta.write(root_index, root_state)

    # ------------------------------------------------------------------ #
    # Preorder traversal: skip root (index 0 in preorder_indices)         #
    # ------------------------------------------------------------------ #
    for step in tf.range(1, node_count):
        node = preorder_indices[step]
        parent = parent_indices[node]
        parent_state = states_ta.read(parent)  # [n, ...batch]

        # Branch transition probs for this node: [...batch, state, state]
        branch_probs = tf.gather(transition_probs, node, axis=-3)

        # Broadcast to [n, ...batch, state, state] then flatten to [N, state, state]
        branch_probs_b = tf.broadcast_to(
            branch_probs,
            tf.concat([sample_and_batch_shape, [state_count, state_count]], axis=0),
        )
        branch_probs_flat = tf.reshape(
            branch_probs_b, [flat_size, state_count, state_count]
        )  # [N, state, state]

        # Select row for parent state: child_probs[i] = branch_probs_flat[i, parent_state_flat[i], :]
        parent_state_flat = tf.reshape(parent_state, [flat_size])  # [N]
        child_probs_flat = tf.gather(
            branch_probs_flat, parent_state_flat, axis=1, batch_dims=1
        )  # [N, state]

        child_state_flat = tf.cast(
            tf.random.stateless_categorical(
                tf.math.log(child_probs_flat), 1, seed=all_seeds[step]
            )[:, 0],
            tf.int32,
        )  # [N]
        child_state = tf.reshape(child_state_flat, sample_and_batch_shape)

        states_ta = states_ta.write(node, child_state)

    # ------------------------------------------------------------------ #
    # Collect leaf states and one-hot encode                              #
    # ------------------------------------------------------------------ #
    # states_ta.gather gives [taxon_count, n, ...batch]
    leaf_states = states_ta.gather(tf.range(taxon_count))

    # Transpose to [n, ...batch, taxon_count]
    n_dims = tf.size(sample_and_batch_shape)
    perm = tf.concat([tf.range(1, n_dims + 1), [0]], axis=0)
    leaf_states = tf.transpose(leaf_states, perm)

    return tf.one_hot(leaf_states, state_count, dtype=frequencies.dtype)


__all__ = ["sample_ctmc_preorder"]
