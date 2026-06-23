import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)


class CompositeConvergenceCriterion(ConvergenceCriterion):
    """Combine several convergence criteria into one.

    Runs a collection of sub-criteria in lockstep and reduces their individual
    ``has_converged`` signals into a single one, according to ``mode``:

    - ``mode="all"`` (the default): converged once **every** sub-criterion has
      converged (logical AND).  Use this to require multiple independent signals
      to agree before stopping, e.g. both the ELBO and the parameters having
      settled.
    - ``mode="any"``: converged as soon as **any** sub-criterion converges
      (logical OR).  Use this for early-out behaviour, e.g. stopping either when
      the loss plateaus or when a non-finite value appears.

    Each sub-criterion keeps its own auxiliary state and applies its own
    ``min_num_steps`` (they are driven via their public ``one_step``), so they can
    be configured independently.  The composite's own ``min_num_steps`` gates the
    combined result on top of that.

    Example:
        >>> criterion = CompositeConvergenceCriterion(
        ...     [RelativeLossNotDecreasing(rtol=1e-6),
        ...      ParametersNotChanging(rtol=1e-5)],
        ...     mode="all",
        ... )
    """

    def __init__(self, criteria, mode="all", min_num_steps=0, name=None):
        """
        Args:
            criteria: non-empty sequence of :class:`ConvergenceCriterion`
                instances to combine.
            mode: ``"all"`` to require every sub-criterion to converge, or
                ``"any"`` to stop as soon as one does.
            min_num_steps: minimum steps before the composite can declare
                convergence (applied on top of each sub-criterion's own
                ``min_num_steps``).
            name: optional name for TF ops.
        """
        criteria = list(criteria)
        if not criteria:
            raise ValueError("`criteria` must be a non-empty sequence.")
        if mode not in ("all", "any"):
            raise ValueError(f"`mode` must be 'all' or 'any', got {mode!r}.")
        self._criteria = criteria
        self._mode = mode
        super().__init__(
            min_num_steps=min_num_steps,
            name=name or "CompositeConvergenceCriterion",
        )

    @property
    def criteria(self):
        return self._criteria

    @property
    def mode(self):
        return self._mode

    def _bootstrap(self, loss, grads, parameters):
        # Drive sub-criteria through their public API so each applies its own
        # name scope / conversions consistently with standalone use.
        return tuple(
            c.bootstrap(loss, grads, parameters) for c in self._criteria
        )

    def _one_step(self, step, loss, grads, parameters, auxiliary_state):
        results = [
            c.one_step(step, loss, grads, parameters, sub_state)
            for c, sub_state in zip(self._criteria, auxiliary_state)
        ]
        has_converged_list = [r[0] for r in results]
        new_state = tuple(r[1] for r in results)

        combined = has_converged_list[0]
        for has_converged in has_converged_list[1:]:
            if self._mode == "all":
                combined = combined & has_converged
            else:
                combined = combined | has_converged

        return combined, new_state


__all__ = ["CompositeConvergenceCriterion"]
