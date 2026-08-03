"""The nonlinear layer of the tree normalising flow.

A monotone, strictly elementwise transform of the per-node coordinates. It is
deliberately free of any tree traversal: all of the flow's structure lives in
the affine tree maps that sandwich it (see
:mod:`treeflow.traversal.tree_affine`), and this layer's only job is to bend
each coordinate so that the composition can represent something other than a
Gaussian.

Parameterisation. Each node gets its own affine pre-conditioning -- two free
parameters, ``scale`` and ``shift`` -- which positions that node's coordinate
within a nonlinearity **whose own parameters are shared across the whole tree**.
So the parameter count is ``2 * internal_node + O(1)`` rather than a full flow's
worth of parameters per node: a node's individual freedom is where it sits on
the shared nonlinearity, not a private copy of it. (Per-node nonlinearity
parameters are still possible -- they simply broadcast -- but they are not the
default, and are not what the flow builds.)

Three nonlinearities are available, all monotone increasing, all analytically
invertible (no root finding), and all the identity at zero raw parameters:

``"spline"`` (default)
    A monotone rational-quadratic spline on ``[-bounds, bounds]``, the identity
    outside it. Very flexible within its window, and its linear tails keep the
    inverse and the log-det-Jacobian well conditioned however the knots move.
``"sinh_arcsinh"``
    ``sinh(tailweight * (arcsinh(x) + skewness))``: unbounded, so unlike the
    spline it reshapes the **tails** -- ``tailweight > 1`` gives heavier tails
    than the base Gaussian, ``< 1`` lighter -- at the cost of only two shared
    parameters and much less freedom in the bulk.
``"affine"``
    No nonlinearity at all: just the per-node affine. The flow then reduces to a
    composition of triangular affine maps -- a Gaussian with tree-structured
    covariance, parameterised in ``O(internal_node)`` rather than
    ``O(internal_node^2)``. This is the ablation that says how much of the
    flow's benefit comes from the nonlinearity rather than from the tree
    structure, and a useful family in its own right.

On tails and the transforms downstream. The flow's coordinates are pushed
through a sigmoid (the non-root height ratios) or an exponential (the root
height) before the ratio transform, so linear tails in flow space mean
logit-normal ratios and a lognormal root -- the same tail behaviour the existing
mean-field and full-rank approximations have. The spline therefore does not make
the tails *worse*; it leaves them alone and reshapes the bulk, where its window
sits, and each node's learnable affine pre-conditioning is what aligns that
window with the node's own bulk. Reshaping the tails is what ``"sinh_arcsinh"``
is for, and the two can be stacked by using more than one flow layer.
"""

import typing as tp

import tensorflow as tf
from tensorflow_probability.python.bijectors import (
    Bijector,
    Identity,
    RationalQuadraticSpline,
    SinhArcsinh,
)
from tensorflow_probability.python.math import softplus_inverse

DEFAULT_NUM_BINS = 8
DEFAULT_BOUNDS = 3.0
DEFAULT_MIN_BIN_GAP = 1e-2
DEFAULT_MIN_KNOT_SLOPE = 1e-2
DEFAULT_MIN_TAILWEIGHT = 1e-2

NONLINEARITIES = ("spline", "sinh_arcsinh", "affine")
DEFAULT_NONLINEARITY = "spline"

#: Raw parameters of each nonlinearity, and their shapes (given ``num_bins``).
#: All are initialised to zeros, at which every nonlinearity is the identity.
_PARAMETER_SHAPES: tp.Dict[str, tp.Callable[[int], tp.Dict[str, tp.Tuple[int, ...]]]] = {
    "spline": lambda num_bins: dict(
        bin_widths=(num_bins,),
        bin_heights=(num_bins,),
        knot_slopes=(num_bins - 1,),
    ),
    "sinh_arcsinh": lambda num_bins: dict(skewness=(), tailweight=()),
    "affine": lambda num_bins: {},
}


def check_nonlinearity(nonlinearity: str) -> str:
    if nonlinearity not in NONLINEARITIES:
        raise ValueError(
            f"nonlinearity must be one of {NONLINEARITIES}; got {nonlinearity!r}"
        )
    return nonlinearity


def shared_parameter_shapes(
    nonlinearity: str = DEFAULT_NONLINEARITY, num_bins: int = DEFAULT_NUM_BINS
) -> tp.Dict[str, tp.Tuple[int, ...]]:
    """Names and shapes of a nonlinearity's shared raw parameters."""
    return _PARAMETER_SHAPES[check_nonlinearity(nonlinearity)](num_bins)


def identity_parameters(
    nonlinearity: str = DEFAULT_NONLINEARITY,
    num_bins: int = DEFAULT_NUM_BINS,
    batch_shape: tp.Sequence[int] = (),
    dtype=tf.float64,
) -> tp.Dict[str, tf.Tensor]:
    """Raw parameters (all zeros) for which the nonlinearity is the identity."""
    return {
        name: tf.zeros(tuple(batch_shape) + shape, dtype=dtype)
        for name, shape in shared_parameter_shapes(nonlinearity, num_bins).items()
    }


def _constrained_bins(raw: tf.Tensor, total: float, min_gap: float) -> tf.Tensor:
    """Positive bin sizes summing to ``total``, each at least ``min_gap``."""
    num_bins = tf.cast(tf.shape(raw)[-1], raw.dtype)
    total = tf.cast(total, raw.dtype)
    min_gap = tf.cast(min_gap, raw.dtype)
    return min_gap + (total - num_bins * min_gap) * tf.math.softmax(raw, axis=-1)


def _positive_from_zero(raw: tf.Tensor, target: float, minimum: float) -> tf.Tensor:
    """Positive value that equals ``target`` when ``raw`` is zero."""
    dtype = raw.dtype
    offset = tf.cast(
        softplus_inverse(tf.constant(target - minimum, dtype=dtype)), dtype
    )
    return tf.cast(minimum, dtype) + tf.math.softplus(raw + offset)


def build_spline(
    parameters: tp.Dict[str, tf.Tensor],
    bounds: float = DEFAULT_BOUNDS,
    min_bin_gap: float = DEFAULT_MIN_BIN_GAP,
    min_knot_slope: float = DEFAULT_MIN_KNOT_SLOPE,
    validate_args: bool = False,
) -> RationalQuadraticSpline:
    """Monotone spline on ``[-bounds, bounds]``, the identity outside it.

    Zero raw parameters give evenly spaced knots with unit slopes -- the
    identity map.
    """
    dtype = parameters["bin_widths"].dtype
    total = 2.0 * bounds
    return RationalQuadraticSpline(
        bin_widths=_constrained_bins(parameters["bin_widths"], total, min_bin_gap),
        bin_heights=_constrained_bins(parameters["bin_heights"], total, min_bin_gap),
        knot_slopes=_positive_from_zero(
            parameters["knot_slopes"], 1.0, min_knot_slope
        ),
        range_min=tf.cast(-bounds, dtype),
        validate_args=validate_args,
    )


def build_nonlinearity(
    nonlinearity: str,
    parameters: tp.Dict[str, tf.Tensor],
    bounds: float = DEFAULT_BOUNDS,
    min_bin_gap: float = DEFAULT_MIN_BIN_GAP,
    min_knot_slope: float = DEFAULT_MIN_KNOT_SLOPE,
    min_tailweight: float = DEFAULT_MIN_TAILWEIGHT,
    validate_args: bool = False,
) -> Bijector:
    """Build an elementwise monotone bijector from its shared raw parameters."""
    check_nonlinearity(nonlinearity)
    if nonlinearity == "affine":
        return Identity()
    if nonlinearity == "sinh_arcsinh":
        return SinhArcsinh(
            skewness=parameters["skewness"],
            tailweight=_positive_from_zero(
                parameters["tailweight"], 1.0, min_tailweight
            ),
            validate_args=validate_args,
        )
    return build_spline(
        parameters,
        bounds=bounds,
        min_bin_gap=min_bin_gap,
        min_knot_slope=min_knot_slope,
        validate_args=validate_args,
    )


class ElementwiseNodeFlow(Bijector):
    """Per-node affine conditioning composed with a shared nonlinearity.

    ``forward(x)[..., i] = nonlinearity(scale[i] * x[..., i] + shift[i])``

    Parameters
    ----------
    scale, shift
        Per-node affine parameters with shape ``[..., internal_node]``.
        ``scale`` must be positive (the flow produces it with a softplus).
    nonlinearity
        Any elementwise monotone increasing bijector -- see
        :func:`build_nonlinearity`. Defaults to the identity, which makes this
        layer a plain per-node affine.
    """

    def __init__(
        self,
        scale: tf.Tensor,
        shift: tf.Tensor,
        nonlinearity: tp.Optional[Bijector] = None,
        name: str = "ElementwiseNodeFlow",
        validate_args: bool = False,
    ):
        parameters = dict(locals())
        self._scale = scale
        self._shift = shift
        self._nonlinearity = Identity() if nonlinearity is None else nonlinearity
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            dtype=scale.dtype,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def nonlinearity(self) -> Bijector:
        return self._nonlinearity

    def _forward(self, x):
        return self._nonlinearity.forward(self._scale * x + self._shift)

    def _inverse(self, y):
        return (self._nonlinearity.inverse(y) - self._shift) / self._scale

    def _forward_log_det_jacobian(self, x):
        affine = self._scale * x + self._shift
        nonlinearity_ldj = self._nonlinearity.forward_log_det_jacobian(
            affine, event_ndims=0
        )
        return tf.reduce_sum(nonlinearity_ldj + tf.math.log(self._scale), axis=-1)

    def _inverse_log_det_jacobian(self, y):
        nonlinearity_ldj = self._nonlinearity.inverse_log_det_jacobian(
            y, event_ndims=0
        )
        return tf.reduce_sum(nonlinearity_ldj - tf.math.log(self._scale), axis=-1)


__all__ = [
    "ElementwiseNodeFlow",
    "NONLINEARITIES",
    "DEFAULT_NONLINEARITY",
    "shared_parameter_shapes",
    "identity_parameters",
    "build_nonlinearity",
    "build_spline",
]
