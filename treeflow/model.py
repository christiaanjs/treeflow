import treeflow.tree_processing
import treeflow.sequences
import treeflow.tree_transform
import treeflow.tf_util
import treeflow.coalescent
import treeflow.substitution_model
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def construct_model_likelihood(newick_file, fasta_file):
    tree, taxon_names = treeflow.tree_processing.parse_newick(newick_file)
    topology = treeflow.tree_processing.update_topology_dict(tree['topology'])
    taxon_count = len(taxon_names)

    sampling_times = tf.convert_to_tensor(tree['heights'][:taxon_count], dtype=tf.float32)

    prior = tfd.JointDistributionNamed(dict(
        frequencies=tfd.Dirichlet(concentration=[4,4,4,4]),
        kappa=tfd.LogNormal(loc=0, scale=1),
        pop_size=tfd.LogNormal(loc=0, scale=1),
        #site_alpha=tfd.LogNormal(loc=0, scale=1),
        #clock_rate=tfd.LogNormal(loc=0, scale=1),
        tree=lambda pop_size: treeflow.coalescent.ConstantCoalescent(taxon_count=taxon_count, pop_size=pop_size, sampling_times=sampling_times)
    ))

    subst_model = treeflow.substitution_model.HKY()
    category_weights = tf.ones(1)
    category_rates = tf.ones(1)
    mutation_rate = tf.convert_to_tensor(0.001)

    alignment = treeflow.sequences.get_encoded_sequences(fasta_file, taxon_names)
    log_prob_conditioned = treeflow.sequences.log_prob_conditioned(alignment, tree['topology'], 1)
                    
    def log_likelihood(tree, kappa, frequencies):  
        return log_prob_conditioned(
            subst_model=subst_model,
            category_weights=category_weights,
            category_rates=category_rates,
            branch_lengths=treeflow.sequences.get_branch_lengths(tree) * mutation_rate,
            frequencies=frequencies,
            kappa=kappa
        )
    log_prob = lambda **z: prior.log_prob(z) + log_likelihood(z['tree'], z['kappa'], z['frequencies'])
    return log_prob, prior


def construct_surrogate_posterior(newick_file):
    tree, taxon_names = treeflow.tree_processing.parse_newick(newick_file)
    topology = treeflow.tree_processing.update_topology_dict(tree['topology'])
    taxon_count = len(taxon_names)
    anchor_heights = treeflow.tree_processing.get_node_anchor_heights(tree['heights'], topology['postorder_node_indices'], topology['child_indices'])
    anchor_heights = tf.convert_to_tensor(anchor_heights, dtype=tf.float32)
    tree_chain = treeflow.tree_transform.TreeChain(
        topology['parent_indices'][taxon_count:] - taxon_count,
        topology['preorder_node_indices'][1:] - taxon_count,
        anchor_heights=anchor_heights)
    init_heights = tf.convert_to_tensor(tree['heights'][taxon_count:], dtype=tf.float32)
    init_heights_trans = tree_chain.inverse(init_heights)
    leaf_heights = tf.convert_to_tensor(tree['heights'][:taxon_count], dtype=tf.float32)

    height_dist = tfd.Blockwise([
        tfd.Independent(tfd.Deterministic(leaf_heights), reinterpreted_batch_ndims=1),
        tfd.TransformedDistribution(
            distribution=tfd.Independent(tfd.Normal(
                loc=tf.Variable(init_heights_trans, name='q_tree_loc'),
                scale=tfp.util.DeferredTensor(tf.Variable(tf.ones_like(init_heights_trans), name='q_tree_scale'), tf.nn.softplus)
            ), reinterpreted_batch_ndims=1),
            bijector=tree_chain
        )
    ])

    q = tfd.JointDistributionNamed(dict(
        tree=treeflow.tree_transform.FixedTopologyDistribution(
            height_distribution=height_dist,
            topology=tree['topology']
        ),
        kappa=tfd.LogNormal(
            loc=tf.Variable(0.0, name='q_kappa_loc'),
            scale=tfp.util.DeferredTensor(tf.Variable(1.0, name='q_kappa_scale'), tf.nn.softplus)
        ),
        pop_size=tfd.LogNormal(
            loc=tf.Variable(0.0, name='q_pop_size_loc'),
            scale=tfp.util.DeferredTensor(tf.Variable(1.0, name='q_pop_size_scale'), tf.nn.softplus)
        ),
        frequencies=tfd.Dirichlet(concentration=tfp.util.DeferredTensor(tf.Variable([4.0, 4.0, 4.0, 4.0], name='q_frequencies_concentration'), tf.nn.softplus))
    ))
    return q
