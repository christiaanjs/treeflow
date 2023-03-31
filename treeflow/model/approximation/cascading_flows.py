from functools import partial
import typing as tp
import tensorflow as tf
from tensorflow_probability.python.bijectors import Sigmoid, Softplus, Chain
from tensorflow_probability.python.distributions import (
    Normal,
    Sample,
    TransformedDistribution,
)
from tensorflow_probability.python.experimental.util import DeferredModule
from tensorflow_probability.python.math import fill_triangular
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.bijectors.highway_flow import HighwayFlowParameters
from treeflow.bijectors.highway_flow_node_bijector import HighwayFlowNodeBijector
from treeflow.bijectors.fixed_topology_bijector import FixedTopologyRootedTreeBijector
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioChainBijector
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor


def build_highway_U(U_diag_inv_softplus: tf.Tensor, U_offdiag: tf.Tensor) -> tf.Tensor:
    offdiag_triu = fill_triangular(U_offdiag, upper=True)
    batch_rank = tf.rank(U_offdiag) - 1
    paddings = tf.concat(
        [tf.zeros((batch_rank, 2), dtype=tf.int32), [[0, 1], [1, 0]]], axis=0
    )
    without_diag = tf.pad(offdiag_triu, paddings)
    diag = Softplus().forward(U_diag_inv_softplus)
    U = tf.linalg.set_diag(without_diag, diag)
    return U


def build_highway_L(L_offdiag: tf.Tensor) -> tf.Tensor:
    offdiag_triu = fill_triangular(L_offdiag, upper=True)
    batch_shape = tf.shape(L_offdiag)[:-1]
    batch_rank = tf.shape(batch_shape)[0]
    k = tf.shape(offdiag_triu)[-1]
    paddings = tf.concat(
        [tf.zeros((batch_rank, 2), dtype=tf.int32), [[1, 0], [0, 1]]], axis=0
    )
    without_diag = tf.pad(offdiag_triu, paddings)
    diag = tf.ones(tf.concat([batch_shape, [k]], axis=0), dtype=L_offdiag.dtype)
    L = tf.linalg.set_diag(without_diag, diag)
    return L


def build_lambd(lambd_logit: tf.Tensor, k: int, aux_k: tp.Optional[int] = None):
    lambd = Sigmoid().forward(lambd_logit)
    if aux_k is None:
        return lambd
    else:
        # Don't gate auxiliary variables
        aux_lambd_shape = tf.concat([tf.shape(lambd_logit)[:-1], [aux_k]], axis=0)
        return tf.concat([lambd, tf.zeros(aux_lambd_shape, dtype=lambd.dtype)], axis=-1)


def get_trainable_highway_flow_parameters(
    k: int,
    batch_shape: tf.Tensor = (),
    aux_k: tp.Optional[int] = None,
    prefix="",
    lambd_init=1.0 - 1e-16,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    kernel_initializer: tp.Optional[tf.keras.initializers.Initializer] = None,
    bias_initializer: tp.Optional[tf.keras.initializers.Initializer] = None,
) -> HighwayFlowParameters:
    batch_and_k = tf.concat([batch_shape, [k]], axis=0)
    if aux_k is None:
        lambd_shape = batch_and_k
    else:
        lambd_shape = tf.concat([batch_shape, [k - aux_k]], axis=0)

    lambd_tensor = tf.broadcast_to(
        tf.convert_to_tensor(lambd_init, dtype=dtype), lambd_shape
    )
    lambd_logit = tf.Variable(
        Sigmoid().inverse(lambd_tensor), name=f"{prefix}lambd_logit"
    )
    lambd = DeferredModule(partial(build_lambd, k=k, aux_k=aux_k), lambd_logit)

    if kernel_initializer is None:
        kernel_initializer = tf.keras.initializers.RandomNormal()
    if bias_initializer is None:
        bias_initializer = tf.keras.initializers.Zeros()
    # U is an upper triangular matrix with positive values on the diagonal
    U_diag_inv_softplus = tf.Variable(
        kernel_initializer(batch_and_k, dtype), name=f"{prefix}U_diag_inv_softplus"
    )
    batch_and_offdiag_size = tf.concat([batch_shape, [k * (k - 1) // 2]])
    U_offdiag = tf.Variable(
        kernel_initializer(batch_and_offdiag_size, dtype), name=f"{prefix}U_offdiag"
    )
    U = DeferredModule(build_highway_U, U_diag_inv_softplus, U_offdiag)
    bias_U = tf.Variable(bias_initializer(batch_and_k, dtype), name=f"{prefix}U_bias")

    # L is a lower triangular matrix with ones on the diagonal
    L_offdiag = tf.Variable(
        kernel_initializer(batch_and_offdiag_size, dtype), name=f"{prefix}L_offdiag"
    )
    L = DeferredModule(build_highway_L, L_offdiag)
    bias_L = tf.Variable(bias_initializer(batch_and_k, dtype), name=f"{prefix}L_bias")
    return HighwayFlowParameters(U=U, bias_U=bias_U, L=L, bias_L=bias_L, lambd=lambd)


def get_cascading_flows_tree_approximation(
    tree: TensorflowRootedTree, name="tree", activation_fn=Sigmoid(), **kwargs
) -> None:
    dtype = tree.node_heights.dtype
    batch_shape = tf.expand_dims(tree.topology.taxon_count - 1, 0)
    parameters = get_trainable_highway_flow_parameters(
        2, batch_shape, prefix=f"{name}_", **kwargs
    )

    flow_bijector = HighwayFlowNodeBijector(
        tree.topology,
        parameters,
        (),
        activation_fn,
    )
    tree_bijector = FixedTopologyRootedTreeBijector(
        tree.topology,
        NodeHeightRatioChainBijector(
            tree.topology, get_anchor_heights_tensor(tree.topology, tree.sampling_times)
        ),
        sampling_times=tree.sampling_times,
    )
    chain_bijector = Chain([tree_bijector, flow_bijector])
    base_dist = Sample(
        Normal(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)),
        batch_shape,
    )
    return TransformedDistribution(base_dist, chain_bijector)
