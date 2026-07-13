import pytest
import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)
from treeflow.vi.convergence_criteria.composite import (
    CompositeConvergenceCriterion,
)


class _ConvergesAtStep(ConvergenceCriterion):
    """Test double that converges exactly when ``step >= target``."""

    def __init__(self, target, name=None):
        self._target = target
        super().__init__(min_num_steps=0, name=name or "ConvergesAtStep")

    def _bootstrap(self, loss, grads, parameters):
        return ()

    def _one_step(self, step, loss, grads, parameters, auxiliary_state):
        has_converged = step >= tf.cast(self._target, step.dtype)
        return has_converged, ()


def _run(criterion, num_steps):
    loss = tf.constant(1.0, dtype=tf.float64)
    state = criterion.bootstrap(loss, grads=None, parameters=[])
    converged_at = None
    for step in range(1, num_steps + 1):
        has_converged, state = criterion.one_step(
            step, loss, grads=None, parameters=[], auxiliary_state=state
        )
        if has_converged.numpy() and converged_at is None:
            converged_at = step
    return converged_at


class TestConstruction:
    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CompositeConvergenceCriterion([])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            CompositeConvergenceCriterion([_ConvergesAtStep(1)], mode="some")


class TestAllMode:
    def test_fires_when_last_subcriterion_converges(self):
        c = CompositeConvergenceCriterion(
            [_ConvergesAtStep(3), _ConvergesAtStep(7)], mode="all"
        )
        assert _run(c, 20) == 7

    def test_does_not_fire_if_one_never_converges(self):
        c = CompositeConvergenceCriterion(
            [_ConvergesAtStep(3), _ConvergesAtStep(1000)], mode="all"
        )
        assert _run(c, 20) is None


class TestAnyMode:
    def test_fires_when_first_subcriterion_converges(self):
        c = CompositeConvergenceCriterion(
            [_ConvergesAtStep(8), _ConvergesAtStep(4)], mode="any"
        )
        assert _run(c, 20) == 4


class TestMinNumSteps:
    def test_composite_min_num_steps_gates_result(self):
        # Both subcriteria converge at step 2, but the composite delays to 10.
        c = CompositeConvergenceCriterion(
            [_ConvergesAtStep(2), _ConvergesAtStep(2)],
            mode="all",
            min_num_steps=10,
        )
        assert _run(c, 20) == 10

    def test_subcriterion_min_num_steps_is_respected(self):
        sub = _ConvergesAtStep(1)
        sub._min_num_steps = tf.convert_to_tensor(5, dtype=tf.int32)
        c = CompositeConvergenceCriterion([sub], mode="all")
        # Sub would converge at step 1 but its own min_num_steps holds it to 5.
        assert _run(c, 20) == 5


class TestStatePropagation:
    def test_state_structure_matches_subcriteria(self):
        from treeflow.vi.convergence_criteria.relative_loss_not_decreasing import (
            RelativeLossNotDecreasing,
        )

        c = CompositeConvergenceCriterion(
            [RelativeLossNotDecreasing(rtol=1e-6), _ConvergesAtStep(3)]
        )
        loss = tf.constant(100.0, dtype=tf.float64)
        state = c.bootstrap(loss, grads=None, parameters=[])
        assert len(state) == 2
        # Sub-state for the loss criterion carries its EWMA forward.
        _, new_state = c.one_step(
            1, tf.constant(90.0, tf.float64), grads=None, parameters=[],
            auxiliary_state=state,
        )
        assert new_state[0].average_decrease.numpy() != 0.0
