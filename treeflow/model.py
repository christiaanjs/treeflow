import treeflow.tree_processing
import treeflow.sequences
import treeflow.tree_transform
import treeflow.tf_util
import treeflow.coalescent
import treeflow.substitution_model
import treeflow.clock_approx
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from treeflow import DEFAULT_FLOAT_DTYPE_TF

distribution_class_supports = {
    tfd.Normal: 'real',
    tfd.LogNormal: 'nonnegative',
    tfd.Beta: '01',
    tfd.Dirichlet: 'simplex'
}

def get_approx_vars(distribution, model_name, dist_name, init_loc=None):
    full_shape = distribution.batch_shape + distribution.event_shape
    init_loc = tf.zeros(full_shape, dtype=distribution.dtype) if init_loc is None else init_loc
    loc_var = tf.Variable(init_loc, name='{0}_{1}_loc'.format(model_name, dist_name))
    init_scale = tf.ones(full_shape, dtype=distribution.dtype)
    scale_var = tf.Variable(tfp.math.softplus_inverse(init_scale), name='{0}_{1}_scale_inv_softplus'.format(model_name, dist_name))
    return  dict(m=loc_var, s=scale_var)

def scale_constraint(var):
    return tfp.util.DeferredTensor(var, tf.nn.softplus)

def construct_distribution_approximation(model_name, dist_name, distribution, init_mode=None, vars=None):
    try:
        support = distribution_class_supports[type(distribution)]
    except KeyError:
        raise ValueError('Distribution not supported: ' + str(distribution))

    if init_mode is not None and vars is not None:
        raise ValueError("Only one of init_mode and vars must be specified")

    base_dist_type = type(distribution.distribution if isinstance(distribution, tfd.Sample) else distribution)
    support = distribution_class_supports[base_dist_type]
    if support == 'real':
        if vars is None:
            vars = get_approx_vars(distribution, model_name, dist_name, init_loc=init_mode)
        batch_dist = tfd.Normal(loc=vars['m'], scale=scale_constraint(vars['s']))
    elif support == 'nonnegative':
        if vars is None:
            vars = get_approx_vars(distribution, model_name, dist_name, init_loc=(None if init_mode is None else (tf.math.log(init_mode) + 1.0)))
        batch_dist = tfd.LogNormal(loc=vars['m'], scale=scale_constraint(vars['s']))
    #elif support == 'simplex': # TODO: Implement Dirichlet
    #    init_concentration = tf.fill(full_shape, tf.convert_to_tensor(2.0, dtype=dtype)) if init_mode is None else (full_shape[-1] * init_mode + 1)
    else:
        raise ValueError('Approximation not yet implemented for support: ' + support)

    event_rank = distribution.event_shape.rank
    if event_rank > 0:
        return tfd.Independent(batch_dist, reinterpreted_batch_ndims=event_rank), vars
    else:
        return batch_dist, vars

def construct_prior_approximation(prior, approx_name='q', init_mode={}, vars={}):
    dists = [(name, dist) for name, dist in prior.model.items() if name not in ['tree', 'rates']]
    res = [(name,) + construct_distribution_approximation(approx_name, name, dist, init_mode.get(name), vars.get(name)) for name, dist in dists]
    names, approx_dists, approx_vars = zip(*res)
    return dict(zip(names, approx_dists)), dict(zip(names, approx_vars))

def construct_tree_approximation(newick_file, approx_name='q', dist_name='tree', approx_model='mean_field', inst=None, vars=None):
    tree, taxon_names = treeflow.tree_processing.parse_newick(newick_file)
    topology = treeflow.tree_processing.update_topology_dict(tree['topology'])
    taxon_count = (tree['heights'].shape[0] + 1) // 2
    anchor_heights = treeflow.tree_processing.get_node_anchor_heights(tree['heights'], topology['postorder_node_indices'], topology['child_indices'])
    anchor_heights = tf.convert_to_tensor(anchor_heights, dtype=DEFAULT_FLOAT_DTYPE_TF)
    tree_chain = treeflow.tree_transform.TreeChain(
        topology['parent_indices'][taxon_count:] - taxon_count,
        topology['preorder_node_indices'][1:] - taxon_count,
        anchor_heights=anchor_heights,
        inst=inst)
    init_heights = tf.convert_to_tensor(tree['heights'][taxon_count:], dtype=DEFAULT_FLOAT_DTYPE_TF)
    init_heights_trans = tree_chain.inverse(init_heights)
    leaf_heights = tf.convert_to_tensor(tree['heights'][:taxon_count], dtype=DEFAULT_FLOAT_DTYPE_TF)

    if approx_model == 'mean_field':
        if vars is None:
            vars = dict(
                m=tf.Variable(init_heights_trans, name='{0}_{1}_loc'.format(approx_name, dist_name)),
                s=tf.Variable(tf.zeros_like(init_heights_trans), name='{0}_{1}_scale_inv_softplus'.format(approx_name, dist_name))
            )
        pretransformed_distribution = tfd.Independent(tfd.Normal(loc=vars['m'], scale=scale_constraint(vars['s'])), reinterpreted_batch_ndims=1)
    else:
        raise ValueError('Approximation not yet implemented for support: ' + support)

    height_dist = treeflow.tree_transform.FixedLeafHeightDistribution(
        tfd.TransformedDistribution(pretransformed_distribution, bijector=tree_chain),
        leaf_heights
    )

    return treeflow.tree_transform.FixedTopologyDistribution(height_distribution=height_dist, topology=tree['topology']), vars

def construct_rate_approximation(rate_dist, approx_name='q', dist_name='rates', approx_model='mean_field', vars=None):
    base_dist, vars = construct_distribution_approximation(approx_name, dist_name, rate_dist, vars=vars) # TODO: Init mode
    if approx_model == 'mean_field':
        final_dist = base_dist
    elif approx_model == 'branch_length_scaled':
        final_dist = lambda tree, clock_rate: treeflow.clock_approx.ScaledRateDistribution(base_dist, tree, clock_rate) # TODO: Does it work having base_dist from closure?
    else:
        raise ValueError('Rate approximation not known: ' + approx_model)
    return final_dist, vars

def get_log_posterior(prior, likelihood, relaxed_clock=False):
    if relaxed_clock:
        return lambda **z: likelihood(treeflow.sequences.get_branch_lengths(z['tree']) * z['clock_rate'] * z['rates']) + prior.log_prob(z)
    else:
        return lambda **z: likelihood(treeflow.sequences.get_branch_lengths(z['tree']) * z['clock_rate']) + prior.log_prob(z)