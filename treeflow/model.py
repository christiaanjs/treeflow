import treeflow.tree_processing
import treeflow.sequences
import treeflow.tree_transform
import treeflow.tf_util
import treeflow.coalescent
import treeflow.substitution_model
import treeflow.clock_approx
import treeflow.priors
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from treeflow import DEFAULT_FLOAT_DTYPE_TF

distribution_class_supports = {
    tfd.Normal: "real",
    tfd.LogNormal: "nonnegative",
    tfd.Gamma: "nonnegative",
    tfd.Beta: "01",
    tfd.Dirichlet: "simplex",
}


def get_normal_approx_vars(distribution, model_name, dist_name, init_loc=None):
    full_shape = distribution.batch_shape + distribution.event_shape
    init_loc = (
        tf.zeros(full_shape, dtype=distribution.dtype) if init_loc is None else init_loc
    )
    loc_var = tf.Variable(init_loc, name="{0}_{1}_loc".format(model_name, dist_name))
    init_scale = tf.ones(full_shape, dtype=distribution.dtype)
    scale_var = tf.Variable(
        tfp.math.softplus_inverse(init_scale),
        name="{0}_{1}_scale_inv_softplus".format(model_name, dist_name),
    )
    return dict(loc=loc_var, scale_inverse_softplus=scale_var)


def scale_constraint(var):
    return tfp.util.DeferredTensor(var, tf.nn.softplus)


def lognormal_loc_mode_match(mode, scale=1.0):
    return tf.math.log(mode) + scale


def get_normal_conjugate_approximation(param, prior_params, input_callable):
    conjugate_dict = treeflow.clock_approx.get_normal_conjugate_posterior_dict(
        **prior_params
    )
    return input_callable(conjugate_dict[param])


def construct_distribution_approximation(
    model_name,
    dist_name,
    distribution,
    init_mode=None,
    vars=None,
    approx=None,
    **approx_kwargs,
):
    if approx is None:
        approx = "mean_field"
    try:
        base_dist_type = type(
            distribution.distribution
            if isinstance(distribution, tfd.Sample)
            else distribution
        )
        support = distribution_class_supports[base_dist_type]
    except KeyError:
        raise ValueError("Distribution not supported: " + str(distribution))

    if init_mode is not None and vars is not None:
        raise ValueError("Only one of init_mode and vars must be specified")

    if approx == "mean_field":
        if support == "real":
            if vars is None:
                vars = get_normal_approx_vars(
                    distribution, model_name, dist_name, init_loc=init_mode
                )
            batch_dist = tfd.Normal(
                loc=vars["loc"], scale=scale_constraint(vars["scale_inverse_softplus"])
            )
        elif support == "nonnegative":
            if vars is None:
                vars = get_normal_approx_vars(
                    distribution,
                    model_name,
                    dist_name,
                    init_loc=(
                        None
                        if init_mode is None
                        else lognormal_loc_mode_match(init_mode)
                    ),
                )
            batch_dist = tfd.LogNormal(
                loc=vars["loc"], scale=scale_constraint(vars["scale_inverse_softplus"])
            )
        # elif support == 'simplex': # TODO: Implement Dirichlet
        #    init_concentration = tf.fill(full_shape, tf.convert_to_tensor(2.0, dtype=dtype)) if init_mode is None else (full_shape[-1] * init_mode + 1)
        else:
            raise ValueError(
                "Mean field approximation not yet implemented for support: " + support
            )
        event_rank = distribution.event_shape.rank
        if event_rank > 0:
            dist = tfd.Independent(batch_dist, reinterpreted_batch_ndims=event_rank)
        else:
            dist = batch_dist
    elif approx == "scaled":
        if support == "nonnegative":
            if vars is None:
                init_statistic = treeflow.clock_approx.get_tree_statistic(
                    **approx_kwargs
                )
                unscaled_init_mode = init_statistic if init_mode is None else None
                vars = get_normal_approx_vars(
                    distribution,
                    model_name,
                    dist_name,
                    init_loc=(
                        None
                        if unscaled_init_mode is None
                        else lognormal_loc_mode_match(unscaled_init_mode)
                    ),
                )
            base_dist_callable = lambda **vars: tfd.LogNormal(
                loc=vars["loc"], scale=scale_constraint(vars["scale_inverse_softplus"])
            )
            approx_kwargs.pop("tree")
            dist = lambda tree: treeflow.clock_approx.ScaledDistribution(
                base_dist_callable, tree=tree, **approx_kwargs, **vars
            )
        else:
            raise ValueError(
                "Scaled approximation only valid for nonnegative support, not "
                + support
            )
    elif approx == "normal_conjugate":  # concentration, rate, loc, precision_scale
        vars = {}
        dist = get_normal_conjugate_approximation(**approx_kwargs)
    return dist, vars


def get_concrete_dist(dist, sample):
    if isinstance(dist, tfp.distributions.Distribution):
        return dist
    else:
        return dist(
            *[
                sample[arg]
                for arg in tfp.distributions.joint_distribution_sequential._get_required_args(
                    dist
                )
            ]
        )


def construct_prior_approximation(
    prior, prior_sample, approx_name="q", init_mode={}, vars={}, approxs={}
):
    dists = [
        (name, get_concrete_dist(dist, prior_sample))
        for name, dist in prior.model.items()
        if name not in ["tree", "rates"]
    ]
    res = [
        (name,)
        + construct_distribution_approximation(
            approx_name,
            name,
            dist,
            init_mode.get(name),
            vars.get(name),
            **(approxs.get(name) or {}),
        )
        for name, dist in dists
    ]
    names, approx_dists, approx_vars = zip(*res)
    return dict(zip(names, approx_dists)), dict(zip(names, approx_vars))


def construct_tree_approximation(
    newick_file,
    approx_name="q",
    dist_name="tree",
    approx_model="mean_field",
    inst=None,
    vars=None,
):
    if inst is None:
        tree, _ = treeflow.tree_processing.parse_newick(newick_file)
    else:
        import treeflow.libsbn

        tree = treeflow.libsbn.get_tree_info(inst).tree
    topology = treeflow.tree_processing.update_topology_dict(tree["topology"])
    taxon_count = (tree["heights"].shape[0] + 1) // 2
    anchor_heights = treeflow.tree_processing.get_node_anchor_heights(
        tree["heights"], topology["postorder_node_indices"], topology["child_indices"]
    )
    anchor_heights = tf.convert_to_tensor(anchor_heights, dtype=DEFAULT_FLOAT_DTYPE_TF)
    tree_chain = treeflow.tree_transform.TreeChain(
        topology["parent_indices"][taxon_count:] - taxon_count,
        topology["preorder_node_indices"][1:] - taxon_count,
        anchor_heights=anchor_heights,
        inst=inst,
    )
    init_heights = tf.convert_to_tensor(
        tree["heights"][taxon_count:], dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    init_heights_trans = tree_chain.inverse(init_heights)
    leaf_heights = tf.convert_to_tensor(
        tree["heights"][:taxon_count], dtype=DEFAULT_FLOAT_DTYPE_TF
    )

    if approx_model == "mean_field":
        if vars is None:
            vars = dict(
                loc=tf.Variable(
                    init_heights_trans,
                    name="{0}_{1}_loc".format(approx_name, dist_name),
                ),
                scale_inverse_softplus=tf.Variable(
                    tf.zeros_like(init_heights_trans),
                    name="{0}_{1}_scale_inv_softplus".format(approx_name, dist_name),
                ),
            )
        pretransformed_distribution = tfd.Independent(
            tfd.Normal(
                loc=vars["loc"], scale=scale_constraint(vars["scale_inverse_softplus"])
            ),
            reinterpreted_batch_ndims=1,
        )
    else:
        raise ValueError(f"Approx model not yet implemented: {approx_model}")

    height_dist = treeflow.tree_transform.FixedLeafHeightDistribution(
        tfd.TransformedDistribution(pretransformed_distribution, bijector=tree_chain),
        leaf_heights,
    )

    return (
        treeflow.tree_transform.FixedTopologyDistribution(
            height_distribution=height_dist, topology=tree["topology"]
        ),
        vars,
    )


def construct_rate_approximation(
    rate_dist,
    prior,
    approx_name="q",
    dist_name="rates",
    approx_model="mean_field",
    vars=None,
):
    if approx_model == "mean_field":

        base_dist, vars = construct_distribution_approximation(
            approx_name, dist_name, rate_dist, vars=vars
        )  # TODO: Init mode
        final_dist = base_dist
    elif approx_model == "scaled":
        base_dist, vars = construct_distribution_approximation(
            approx_name, dist_name, rate_dist, vars=vars
        )
        base_dist_callable = lambda **vars: construct_distribution_approximation(
            approx_name, dist_name, rate_dist, vars=vars
        )[0]
        if "clock_rate" in prior.model:
            final_dist = (
                lambda tree, clock_rate: treeflow.clock_approx.ScaledRateDistribution(
                    base_dist_callable, tree, clock_rate, **vars
                )
            )
        else:
            final_dist = lambda tree: treeflow.clock_approx.ScaledRateDistribution(
                base_dist_callable, tree, **vars
            )
    elif (
        approx_model == "scaled_shrinkage"
    ):  # TODO: Does it make sense to use construct_distribution_approximation here? Is DeferredTensor an issue?

        if vars is None:
            full_shape = rate_dist.batch_shape + rate_dist.event_shape
            vars = {}
            vars["log_r_prior_loc"] = tf.Variable(
                tf.zeros(rate_dist.batch_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_prior_loc".format(approx_name, dist_name),
            )
            vars["log_r_prior_precision_inv_softplus"] = tf.Variable(
                tf.zeros(rate_dist.batch_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_prior_precision_inv_softplus".format(
                    approx_name, dist_name
                ),
            )
            vars["log_d"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_d".format(approx_name, dist_name),
            )
            vars["log_r_precision_inv_softplus"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_precision_inv_softplus".format(
                    approx_name, dist_name
                ),
            )

        def rate_dist(tree):
            log_blens = tf.math.log(treeflow.sequences.get_branch_lengths(tree))
            observations = vars["log_d"] - log_blens
            prior_precision = tf.expand_dims(
                scale_constraint(vars["log_r_prior_precision_inv_softplus"]), -1
            )
            prior_loc = tf.expand_dims(vars["log_r_prior_loc"], -1)
            sampling_precision = scale_constraint(vars["log_r_precision_inv_softplus"])
            posterior_precision = prior_precision + sampling_precision
            posterior_loc = (
                prior_loc * prior_precision + observations * sampling_precision
            ) / posterior_precision
            return tfd.Independent(
                tfd.LogNormal(
                    loc=posterior_loc,
                    scale=treeflow.priors.precision_to_scale(posterior_precision),
                ),
                reinterpreted_batch_ndims=1,
            )

        final_dist = rate_dist

    elif (
        approx_model == "scaled_shrinkage_local"
    ):  # TODO: Does it make sense to use construct_distribution_approximation here? Is DeferredTensor an issue?

        if vars is None:
            full_shape = rate_dist.batch_shape + rate_dist.event_shape
            vars = {}
            vars["log_r_prior_loc"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_prior_loc".format(approx_name, dist_name),
            )
            vars["log_r_prior_precision_inv_softplus"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_prior_precision_inv_softplus".format(
                    approx_name, dist_name
                ),
            )
            vars["log_d"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_d".format(approx_name, dist_name),
            )
            vars["log_r_precision_inv_softplus"] = tf.Variable(
                tf.zeros(full_shape, dtype=rate_dist.dtype),
                name="{0}_{1}_log_r_precision_inv_softplus".format(
                    approx_name, dist_name
                ),
            )

        def rate_dist(tree):
            log_blens = tf.math.log(treeflow.sequences.get_branch_lengths(tree))
            observations = vars["log_d"] - log_blens
            prior_precision = scale_constraint(
                vars["log_r_prior_precision_inv_softplus"]
            )
            prior_loc = vars["log_r_prior_loc"]
            sampling_precision = scale_constraint(vars["log_r_precision_inv_softplus"])
            posterior_precision = prior_precision + sampling_precision
            posterior_loc = (
                prior_loc * prior_precision + observations * sampling_precision
            ) / posterior_precision
            return tfd.Independent(
                tfd.LogNormal(
                    loc=posterior_loc,
                    scale=treeflow.priors.precision_to_scale(posterior_precision),
                ),
                reinterpreted_batch_ndims=1,
            )

        final_dist = rate_dist

    #     if "clock_rate" in prior.model:
    #         final_dist = (
    #             lambda tree, clock_rate: treeflow.clock_approx.ScaledRateDistribution(
    #                 base_dist_callable, tree, clock_rate, **vars
    #             )
    #         )
    #     else:
    #         final_dist = lambda tree: treeflow.clock_approx.ScaledRateDistribution(
    #             base_dist_callable, tree, **vars
    #         )
    else:
        raise ValueError("Rate approximation not known: " + approx_model)
    return final_dist, vars


def get_log_posterior(prior, likelihood):
    if "rates" in prior.model and "clock_rate" in prior.model:
        blen_func = (
            lambda **z: treeflow.sequences.get_branch_lengths(z["tree"])
            * z["clock_rate"]
            * z["rates"]
        )
    elif "rates" in prior.model:
        blen_func = (
            lambda **z: treeflow.sequences.get_branch_lengths(z["tree"]) * z["rates"]
        )
    elif "clock_rate" in prior.model:
        blen_func = (
            lambda **z: treeflow.sequences.get_branch_lengths(z["tree"])
            * z["clock_rate"]
        )
    else:
        raise ValueError(
            "One or both of clock_rate and rates must be modelled in the prior"
        )

    return lambda **z: likelihood(blen_func(**z)) + prior.log_prob(z)
