import numpy as np
import pytest
import tensorflow as tf
from treeflow.vi.convergence_criteria.relative_loss_not_decreasing import (
    RelativeLossNotDecreasing,
)


def _run(criterion, losses):
    """Drive the criterion on a list of scalar loss values; return per-step state.

    Uses the public bootstrap/one_step API so that min_num_steps is applied.
    grads=None and parameters=[] match what the public wrapper accepts when
    those quantities are unused by the criterion.
    """
    losses_tf = [tf.constant(l, dtype=tf.float64) for l in losses]
    state = criterion.bootstrap(losses_tf[0], grads=None, parameters=[])
    states = []
    converged_at = None
    for step, loss in enumerate(losses_tf[1:], start=1):
        has_converged, state = criterion.one_step(step, loss, grads=None, parameters=[], auxiliary_state=state)
        states.append(state)
        if has_converged.numpy() and converged_at is None:
            converged_at = step
    return states, converged_at


class TestBootstrap:
    def test_initial_ewma_zero(self):
        c = RelativeLossNotDecreasing(rtol=1e-4)
        state = c._bootstrap(tf.constant(100.0, tf.float64), None, None)
        assert state.average_decrease.numpy() == pytest.approx(0.0)

    def test_initial_rel_rate_zero(self):
        c = RelativeLossNotDecreasing(rtol=1e-4)
        state = c._bootstrap(tf.constant(100.0, tf.float64), None, None)
        assert state.rel_rate.numpy() == pytest.approx(0.0)

    def test_previous_loss_set(self):
        c = RelativeLossNotDecreasing(rtol=1e-4)
        state = c._bootstrap(tf.constant(42.0, tf.float64), None, None)
        assert state.previous_loss.numpy() == pytest.approx(42.0)

    def test_initial_consecutive_below_zero(self):
        c = RelativeLossNotDecreasing(rtol=1e-4)
        state = c._bootstrap(tf.constant(100.0, tf.float64), None, None)
        assert state.consecutive_below.numpy() == 0


class TestEWMAUpdate:
    def test_first_step_ewma_equals_decrease_over_window(self):
        """ewma[1] = decrease / W  (bootstrap initialises ewma=0)."""
        W = 10
        c = RelativeLossNotDecreasing(rtol=1e-4, window_size=W)
        losses = [100.0, 90.0]
        states, _ = _run(c, losses)
        expected = (100.0 - 90.0) / W
        assert states[0].average_decrease.numpy() == pytest.approx(expected)

    def test_ewma_decays_on_flat_loss(self):
        """When improvement = 0 every step, ewma decays as decay^t * ewma_0."""
        W = 5
        c = RelativeLossNotDecreasing(rtol=1e-6, window_size=W)
        # One big improvement then plateau
        losses = [100.0, 50.0] + [50.0] * 20
        states, _ = _run(c, losses)
        decay = 1.0 - 1.0 / W
        ewma_after_first = (100.0 - 50.0) / W
        for i, state in enumerate(states[1:], start=1):
            expected = ewma_after_first * (decay ** i)
            assert state.average_decrease.numpy() == pytest.approx(expected, rel=1e-9)

    def test_rel_rate_equals_ewma_over_abs_loss(self):
        W = 10
        c = RelativeLossNotDecreasing(rtol=1e-4, window_size=W)
        losses = [200.0, 180.0, 160.0, 150.0]
        states, _ = _run(c, losses)
        for state, loss in zip(states, losses[1:]):
            expected_rel = state.average_decrease.numpy() / abs(loss)
            assert state.rel_rate.numpy() == pytest.approx(expected_rel, rel=1e-9)


class TestNaNHandling:
    def test_nan_step_does_not_update_previous_loss(self):
        """previous_loss should be frozen on NaN steps."""
        c = RelativeLossNotDecreasing(rtol=1e-4, window_size=10)
        losses = [100.0, float("nan"), 80.0]
        states, _ = _run(c, losses)
        # After the NaN step, previous_loss should still be 100.0
        # so the decrease on step 2 is 100-80=20, not nan-80
        assert np.isfinite(states[1].average_decrease.numpy())

    def test_nan_step_ewma_decays_via_safe_decrease(self):
        """NaN steps contribute safe_decrease=0, so ewma *= decay."""
        W = 10
        c = RelativeLossNotDecreasing(rtol=1e-4, window_size=W)
        decay = 1.0 - 1.0 / W
        losses = [100.0, 90.0, float("nan")]
        states, _ = _run(c, losses)
        ewma_after_step1 = states[0].average_decrease.numpy()
        ewma_after_nan = states[1].average_decrease.numpy()
        assert ewma_after_nan == pytest.approx(ewma_after_step1 * decay, rel=1e-9)

    def test_convergence_not_declared_on_nan_step(self):
        """Convergence must never fire on the NaN step itself."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, window_size=2)
        losses = [100.0, float("nan")]
        _, converged_at = _run(c, losses)
        assert converged_at is None


class TestConvergence:
    def test_rtol_fires_after_min_steps(self):
        """With a very loose rtol, convergence fires after min_num_steps."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=3, min_consecutive=1, window_size=2)
        losses = [100.0] * 10
        _, converged_at = _run(c, losses)
        assert converged_at is not None
        assert converged_at >= 3

    def test_rtol_does_not_fire_before_min_steps(self):
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=50, min_consecutive=1, window_size=2)
        losses = [100.0] * 10
        _, converged_at = _run(c, losses)
        assert converged_at is None

    def test_negative_ewma_does_not_trigger_rtol(self):
        """When loss is increasing (ewma < 0), rtol must not fire."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, min_consecutive=1, window_size=2)
        # Loss increases every step → ewma will be negative
        losses = [10.0, 20.0, 30.0, 40.0, 50.0]
        _, converged_at = _run(c, losses)
        assert converged_at is None

    def test_atol_fires(self):
        """atol convergence fires when ewma drops below threshold."""
        c = RelativeLossNotDecreasing(atol=0.1, min_num_steps=0, min_consecutive=1, window_size=2)
        # Flat loss → ewma decays to 0, should eventually drop below atol=0.1
        losses = [100.0, 99.0] + [99.0] * 50
        _, converged_at = _run(c, losses)
        assert converged_at is not None

    def test_negative_ewma_does_not_trigger_atol(self):
        """When ewma < 0 (loss worsening), atol must not fire."""
        c = RelativeLossNotDecreasing(atol=1e6, min_num_steps=0, min_consecutive=1, window_size=2)
        losses = [10.0, 20.0, 30.0, 40.0, 50.0]
        _, converged_at = _run(c, losses)
        assert converged_at is None

    def test_must_specify_rtol_or_atol(self):
        with pytest.raises(ValueError, match="rtol"):
            RelativeLossNotDecreasing()

    def test_single_dip_does_not_converge(self):
        """A lone step below rtol must not fire when min_consecutive > 1."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, min_consecutive=3, window_size=2)
        # First two steps: flat (ewma decays toward 0, satisfies loose rtol)
        # Third step: brief spike (loss worsens → ewma goes negative, resets counter)
        # Fourth+ steps: flat again
        losses = [100.0, 100.0, 200.0] + [200.0] * 20
        states, converged_at = _run(c, losses)
        # After the spike at step 2, consecutive_below must have reset to 0
        spike_idx = 1  # 0-based index into states (step 2 is the worsening step)
        assert states[spike_idx].consecutive_below.numpy() == 0

    def test_consecutive_counter_increments(self):
        """Counter increments each step the condition is met."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, min_consecutive=5, window_size=2)
        losses = [100.0] * 10
        states, _ = _run(c, losses)
        for i, state in enumerate(states):
            assert state.consecutive_below.numpy() == i + 1

    def test_fires_after_min_consecutive(self):
        """Convergence fires exactly when counter reaches min_consecutive."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, min_consecutive=4, window_size=2)
        losses = [100.0] * 20
        _, converged_at = _run(c, losses)
        assert converged_at == 4

    def test_nan_resets_consecutive_counter(self):
        """A NaN step resets the consecutive counter."""
        c = RelativeLossNotDecreasing(rtol=1e6, min_num_steps=0, min_consecutive=3, window_size=2)
        # Two satisfying steps, then NaN, then satisfying steps
        losses = [100.0, 100.0, 100.0, float("nan"), 100.0, 100.0, 100.0, 100.0]
        states, _ = _run(c, losses)
        nan_idx = 2  # 0-based into states (step 3 is NaN)
        assert states[nan_idx].consecutive_below.numpy() == 0
