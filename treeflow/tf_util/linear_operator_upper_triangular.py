from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util

__all__ = [
    "LinearOperatorUpperTriangular",
]


@linear_operator.make_composite_tensor
class LinearOperatorUpperTriangular(linear_operator.LinearOperator):
    def __init__(
        self,
        triu,
        is_non_singular=None,
        is_self_adjoint=None,
        is_positive_definite=None,
        is_square=None,
        name="LinearOperatorUpperTriangular",
    ):
        parameters = dict(
            triu=triu,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )

        if is_square is False:
            raise ValueError(
                "Only square lower triangular operators supported at this time."
            )
        is_square = True

        with ops.name_scope(name, values=[triu]):
            self._triu = linear_operator_util.convert_nonref_to_tensor(
                triu, name="tril"
            )
            self._check_triu(self._triu)

            super(LinearOperatorUpperTriangular, self).__init__(
                dtype=self._triu.dtype,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                parameters=parameters,
                name=name,
            )

    @property
    def triu(self):
        """The upper triangular matrix defining this operator."""
        return self._triu

    def _check_triu(self, triu):
        """Static check of the `tril` argument."""

        if triu.shape.ndims is not None and triu.shape.ndims < 2:
            raise ValueError(
                "Argument tril must have at least 2 dimensions.  Found: %s" % triu
            )

    def _get_triu(self):
        """Gets the `tril` kwarg, with upper part zero-d out."""
        return array_ops.matrix_band_part(self._triu, 0, -1)

    def _get_diag(self):
        """Gets the diagonal part of `triu` kwarg."""
        return array_ops.matrix_diag_part(self._triu)

    def _shape(self):
        return self._triu.shape

    def _shape_tensor(self):
        return array_ops.shape(self._triu)

    def _assert_non_singular(self):
        return linear_operator_util.assert_no_entries_with_modulus_zero(
            self._get_diag(),
            message="Singular operator:  Diagonal contained zero values.",
        )

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return math_ops.matmul(
            self._get_triu(), x, adjoint_a=adjoint, adjoint_b=adjoint_arg
        )

    def _determinant(self):
        return math_ops.reduce_prod(self._get_diag(), axis=[-1])

    def _log_abs_determinant(self):
        return math_ops.reduce_sum(
            math_ops.log(math_ops.abs(self._get_diag())), axis=[-1]
        )

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        return linalg.triangular_solve(
            self._get_triu(), rhs, lower=False, adjoint=adjoint
        )

    def _to_dense(self):
        return self._get_triu()

    def _eigvals(self):
        return self._get_diag()

    @property
    def _composite_tensor_fields(self):
        return ("triu",)

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        return {"triu": 2}
