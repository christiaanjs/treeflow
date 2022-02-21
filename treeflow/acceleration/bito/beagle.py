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
    **subst_model_params
) -> tp.Tuple[tp.Callable[[tf.Tensor], tf.Tensor], object]:

    if isinstance(subst_model, treeflow_subst.JC):
        subst_model_string = "JC69"
        param_updates = {}
    elif isinstance(subst_model, treeflow_subst.HKY):
        subst_model_string = "HKY"
        kappa = subst_model_params["kappa"]
        rates = np.array([kappa])
        param_updates = {"substitution model rates": rates, "substitution model frequencies": np.array(frequencies)}
    else:
        raise ValueError("Unsupported substitution model")

    if inst is None:
        if newick_file is None:
            raise ValueError("Either a bito instance or Newick file must be supplied")
        inst = bito_instance.get_instance(newick_file, dated=dated)
    inst.read_fasta_file(fasta_file)
    inst.set_rescaling(rescaling)
    model_specification = bito.PhyloModelSpecification(
        subst_model_string, "constant", "strict"
    )
    inst.prepare_for_phylo_likelihood(model_specification, 1)

    phylo_model_param_block_map = inst.get_phylo_model_param_block_map()
    phylo_model_param_block_map["clock rate"][:] = 1.0

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
