from treeflow.vi.convergence_criteria.composite import (
    CompositeConvergenceCriterion,
)
from treeflow.vi.convergence_criteria.relative_loss_not_decreasing import (
    RelativeLossNotDecreasing,
)
from treeflow.vi.convergence_criteria.parameters_not_changing import (
    ParametersNotChanging,
)


class LossAndParametersNotChanging(CompositeConvergenceCriterion):
    """Require both the loss and the parameters to have stopped moving.

    A convenience wrapper bundling :class:`RelativeLossNotDecreasing` and
    :class:`ParametersNotChanging` into a single criterion that shares one
    ``rtol`` (interpreted relative to ``|loss|`` for the loss term and relative
    to ``||theta||`` for the parameter term).  With the default ``mode="all"``
    optimisation stops only once *both* the ELBO and the parameters have settled,
    which is more robust than either signal alone: a noisy ELBO estimate can keep
    drifting after the parameters have converged, and conversely the parameters
    can still be moving while the ELBO looks flat.

    The shared ``window_size``, ``min_num_steps`` and ``min_consecutive`` are
    applied to both sub-criteria.  Use ``param_atol`` to give the parameter term
    an absolute floor (see Note).

    Note:
        The parameter ``rtol`` divides by ``rms(theta)``, which is fragile when a
        parameter's optimum is at (or near) zero: ``rms(theta)`` shrinks toward 0
        and the relative change never settles.  Supplying ``param_atol`` adds an
        OR-ed absolute test (``ewma < param_atol``) on the smoothed per-coordinate
        RMS update that fires in that regime.  Because the criterion uses a
        per-coordinate RMS (not a raw norm), ``param_atol`` is a per-coordinate
        change tolerance in unconstrained space (e.g. ``1e-3``) and is
        independent of the number of parameters.
    """

    def __init__(
        self,
        rtol,
        param_atol=None,
        mode="all",
        window_size=100,
        min_num_steps=500,
        min_consecutive=10,
        name=None,
    ):
        """
        Args:
            rtol: shared relative convergence threshold, applied to both the loss
                term (relative to ``|loss|``) and the parameter term (relative to
                ``||theta||``).
            param_atol: optional absolute threshold for the parameter term on the
                smoothed update norm.  Recommended when any parameter's optimum
                may be at zero (see Note).
            mode: ``"all"`` (default) to require both terms to converge, or
                ``"any"`` to stop as soon as either does.
            window_size: shared EWMA time-constant (in steps).
            min_num_steps: shared minimum number of steps before convergence.
            min_consecutive: shared number of consecutive satisfying steps
                required before stopping.
            name: optional name for TF ops.
        """
        loss_criterion = RelativeLossNotDecreasing(
            rtol=rtol,
            window_size=window_size,
            min_num_steps=min_num_steps,
            min_consecutive=min_consecutive,
        )
        parameters_criterion = ParametersNotChanging(
            rtol=rtol,
            atol=param_atol,
            window_size=window_size,
            min_num_steps=min_num_steps,
            min_consecutive=min_consecutive,
        )
        self._rtol = rtol
        self._param_atol = param_atol
        self._loss_criterion = loss_criterion
        self._parameters_criterion = parameters_criterion
        super().__init__(
            [loss_criterion, parameters_criterion],
            mode=mode,
            name=name or "LossAndParametersNotChanging",
        )

    @property
    def rtol(self):
        return self._rtol

    @property
    def param_atol(self):
        return self._param_atol

    @property
    def loss_criterion(self):
        return self._loss_criterion

    @property
    def parameters_criterion(self):
        return self._parameters_criterion


__all__ = ["LossAndParametersNotChanging"]
