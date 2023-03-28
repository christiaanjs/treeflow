import typing as tp
import tensorflow as tf
from tensorflow_probability.python.bijectors import Bijector, Chain
from treeflow.tf_util.linear_operator_upper_triangular import (
    LinearOperatorUpperTriangular,
)


class TriangularHighwayLayer(Bijector):
    def __init__(
        self,
        lambd: tf.Tensor,
        tri: tp.Union[
            tf.linalg.LinearOperatorLowerTriangular, LinearOperatorUpperTriangular
        ],
        bias: tf.Tensor,
        validate_args=False,
        name="TriangularHighwayLayer",
    ):
        params = locals()
        self._lambd = lambd
        self._tri = tri
        self._bias = bias
        if isinstance(self._tri, tf.linalg.LinearOperatorLowerTriangular):
            self._lower = True
        elif isinstance(self._tri, LinearOperatorUpperTriangular):
            self._lower = False
        else:
            raise ValueError(f"tri must be upper or lower triangular linear operator")
        super().__init__(
            validate_args=validate_args,
            name=name,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            parameters=params,
            is_constant_jacobian=True,
        )

    def _forward(self, x):
        return self._lambd * x + (1 - self._lambd) * (self._tri.matvec(x) + self._bias)

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(
            tf.math.log(self._lambd + (1 - self._lambd) * self._tri.diag_part())
        )

    def _inverse(self, y):
        rhs = y - (1 - self._lambd) * self._bias
        lhs = (
            self._lambd * tf.eye(y.shape[-1]) + (1 - self._lambd) * self._tri.to_dense()
        )
        lhs_operator = (
            tf.linalg.LinearOperatorLowerTriangular(lhs)
            if self._lower
            else LinearOperatorUpperTriangular(lhs)
        )
        return lhs_operator.solvevec(rhs)


class HighwayActivationLayer(Bijector):
    def __init__(
        self,
        lambd: tf.Tensor,
        activation_function: Bijector,
        validate_args=False,
        name="HighwayActivationLayer",
    ):
        params = locals()
        self._lambd = lambd
        self._activation_function = activation_function
        super().__init__(
            validate_args=validate_args,
            name=name,
            forward_min_event_ndims=self._activation_function.forward_min_event_ndims,
            inverse_min_event_ndims=self._activation_function.inverse_min_event_ndims,
            parameters=params,
        )

    def _forward(self, x):
        return x * self._lambd + (1 - self._lambd) * self._activation_function.forward(
            x
        )

    def _forward_log_det_jacobian(self, x):
        return tf.math.log(
            self._lambd
            + (1 - self._lambd) * self._activation_function.forward_log_det_jacobian(x)
        )


class HighwayFlow(Chain):
    def __init__(
        self,
        lambd,
        U,
        L,
        activation_function,
        validate_args=False,
        name="HighwayFlow",
    ):
        super().__init__(
            [
                HighwayActivationLayer(
                    lambd, activation_function, validate_args=validate_args
                ),
                TriangularHighwayLayer(lambd, LinearOperatorUpperTriangular(U)),
                TriangularHighwayLayer(
                    lambd, tf.linalg.LinearOperatorLowerTriangular(L)
                ),
            ],
            validate_args=validate_args,
            name=name,
        )
