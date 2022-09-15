import bito
import typing as tp
import tensorflow as tf
import treeflow.acceleration.bito.instance as bito_instance
import treeflow.evolution.substitution as treeflow_subst
import numpy as np
from treeflow import DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_NP


def phylogenetic_likelihood(
    fasta_file,
    subst_model,
    frequencies,
    rescaling=False,
    inst=None,
    newick_file=None,
    dated=True,
    clock_rate=1.0,
    site_model="none",
    site_model_params=None,
    **subst_model_params,
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], object]:

    if isinstance(subst_model, treeflow_subst.JC):
        subst_model_string = "JC69"
        param_updates = {}
    elif isinstance(subst_model, treeflow_subst.HKY):
        subst_model_string = "HKY"
        kappa = subst_model_params["kappa"]
        rates = np.array([kappa])
        param_updates = {
            "substitution_model_rates": rates,
            "substitution_model_frequencies": np.array(frequencies),
        }
    elif isinstance(subst_model, treeflow_subst.GTR):
        subst_model_string = "GTR"
        rates = subst_model_params["rates"]
        param_updates = {
            "substitution_model_rates": rates,
            "substitution_model_frequencies": np.array(frequencies),
        }
    else:
        raise ValueError(f"Unsupported substitution model: {subst_model}")

    if site_model == "discrete_weibull":
        param_updates["Weibull shape"] = site_model_params["site_weibull_concentration"]
        category_count = site_model_params["category_count"]
        site_model_string = f"weibull+{category_count}"
    elif site_model == "none":
        site_model_string = "constant"
    else:
        raise ValueError(f"Unsupported site model: {site_model}")

    if inst is None:
        if newick_file is None:
            raise ValueError("Either a bito instance or Newick file must be supplied")
        inst = bito_instance.get_instance(newick_file, dated=dated)
    inst.read_fasta_file(fasta_file)
    inst.set_rescaling(rescaling)
    model_specification = bito.PhyloModelSpecification(
        subst_model_string, site_model_string, "strict"
    )
    inst.prepare_for_phylo_likelihood(model_specification, 1)

    phylo_model_param_block_map = inst.get_phylo_model_param_block_map()
    phylo_model_param_block_map["clock_rate"][:] = clock_rate

    for key, value in param_updates.items():
        phylo_model_param_block_map[key][:] = value

    branch_lengths = np.array(inst.tree_collection.trees[0].branch_lengths, copy=False)

    def bito_func(x):
        """Wrapping likelihood and gradient evaluation via beagle."""
        branch_lengths[:-1] = x
        gradient = inst.phylo_gradients()[0]
        grad_array = np.array(
            gradient.gradient["branch_lengths"], dtype=DEFAULT_FLOAT_DTYPE_NP
        )[:-1]
        return (
            np.array(gradient.log_likelihood, dtype=DEFAULT_FLOAT_DTYPE_NP),
            grad_array,
        )

    bito_func_vec = np.vectorize(
        bito_func,
        [DEFAULT_FLOAT_DTYPE_NP, DEFAULT_FLOAT_DTYPE_NP],
        signature="(n)->(),(n)",
    )

    @tf.custom_gradient
    def bito_tf_func(x):
        logp, grad_val = tf.numpy_function(
            bito_func_vec, [x], [DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_TF]
        )

        def grad(dlogp):
            return tf.expand_dims(dlogp, -1) * grad_val

        return logp, grad

    return bito_tf_func, inst
