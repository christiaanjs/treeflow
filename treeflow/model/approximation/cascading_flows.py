import typing as tp
import attr
import tensorflow as tf
from tensorflow_probability.python.bijectors import Sigmoid, Softplus
from tensorflow_probability.python.experimental.util import DeferredModule
from tensorflow_probability.python.math import fill_triangular
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.bijectors.preorder_node_bijector import PreorderNodeBijector


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


@attr.s(auto_attribs=True)
class HighwayFlowParameters:
    U: tf.Tensor
    bias_U: tf.Tensor
    L: tf.Tensor
    bias_L: tf.Tensor
    lambd: tf.Tensor


def get_trainable_highway_flow_parameters(
    k: int,
    batch_shape: tf.Tensor,
    prefix="",
    lambd_init=1.0 - 1e-16,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    kernel_initializer: tp.Optional[tf.keras.initializers.Initializer] = None,
    bias_initializer: tp.Optional[tf.keras.initializers.Initializer] = None,
) -> HighwayFlowParameters:
    lambd_bijector = Sigmoid()
    lambd_tensor = tf.broadcast_to(
        tf.convert_to_tensor(lambd_init, dtype=dtype), batch_shape
    )
    lambd_logit = tf.Variable(
        lambd_bijector.inverse(lambd_tensor), name=f"{prefix}lambd_logit"
    )
    lambd = DeferredModule(lambd_bijector, lambd_logit)
    if kernel_initializer is None:
        kernel_initializer = tf.keras.initializers.RandomNormal()
    if bias_initializer is None:
        bias_initializer = tf.keras.initializers.Zeros()
    batch_and_k = tf.concat([batch_shape, [k]], axis=0)
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

    L_offdiag = tf.Variable(
        kernel_initializer(batch_and_offdiag_size, dtype), name=f"{prefix}L_offdiag"
    )
    L = DeferredModule(build_highway_L, L_offdiag)
    bias_L = tf.Variable(bias_initializer(batch_and_k, dtype), name=f"{prefix}L_bias")
    return HighwayFlowParameters(U=U, bias_U=bias_U, L=L, bias_L=bias_L, lambd=lambd)


def build_highway_flow_bijector():
    ...


def get_cascading_flows_tree_approximation(
    topology: TensorflowTreeTopology, name="tree", k=3, **kwargs
) -> None:
    batch_shape = tf.expand_dims(topology.taxon_count - 1, 0)
    parameters = get_trainable_highway_flow_parameters(
        k, batch_shape, f"{name}_", **kwargs
    )
