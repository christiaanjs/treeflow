import math
import numpy as np
import pytest
import tensorflow as tf
from treeflow.vi.convergence_criteria.parameters_not_changing import (
    ParametersNotChanging,
)


def _params(values):
    """Build a single-parameter list from a python value."""
    return [tf.constant(values, dtype=tf.float64)]


def _run(criterion, parameter_sequence):
    """Drive the criterion over a sequence of parameter lists.

    Each element of ``parameter_sequence`` is the full parameters list at that
    step.  Loss is held constant (the criterion ignores it).  Returns per-step
    states and the step at which convergence first fired (or ``None``).
    """
    loss = tf.constant(1.0, dtype=tf.float64)
    state = criterion.bootstrap(loss, grads=None, parameters=parameter_sequence[0])
    states = []
    converged_at = None
    for step, params in enumerate(parameter_sequence[1:], start=1):
        has_converged, state = criterion.one_step(
            step, loss, grads=None, parameters=params, auxiliary_state=state
        )
        states.append(state)
        if has_converged.numpy() and converged_at is None:
            converged_at = step
    return states, converged_at


class TestBootstrap:
    def test_initial_ewma_zero(self):
        c = ParametersNotChanging(rtol=1e-4)
        state = c._bootstrap(tf.constant(1.0, tf.float64), None, _params(3.0))
        assert state.average_change.numpy() == pytest.approx(0.0)

    def test_initial_consecutive_zero(self):
        c = ParametersNotChanging(rtol=1e-4)
        state = c._bootstrap(tf.constant(1.0, tf.float64), None, _params(3.0))
        assert state.consecutive_below.numpy() == 0

    def test_previous_parameters_stored(self):
        c = ParametersNotChanging(rtol=1e-4)
        state = c._bootstrap(tf.constant(1.0, tf.float64), None, _params([1.0, 2.0]))
        np.testing.assert_allclose(state.previous_parameters[0].numpy(), [1.0, 2.0])


class TestChangeComputation:
    def test_change_is_per_coordinate_rms(self):
        """ewma[1] = rms(theta1 - theta0) / W (bootstrap ewma=0)."""
        W = 10
        c = ParametersNotChanging(rtol=1e-9, window_size=W)
        # change across two scalar params: rms = sqrt((3^2 + 4^2) / 2) = sqrt(12.5)
        seq = [
            [tf.constant([0.0], tf.float64), tf.constant([0.0], tf.float64)],
            [tf.constant([3.0], tf.float64), tf.constant([4.0], tf.float64)],
        ]
        states, _ = _run(c, seq)
        expected_rms = math.sqrt((3.0**2 + 4.0**2) / 2.0)
        assert states[0].average_change.numpy() == pytest.approx(expected_rms / W)

    def test_atol_independent_of_parameter_count(self):
        """A duplicated parameter set (2x the params) yields the same RMS change."""
        W = 4
        single = ParametersNotChanging(rtol=1e-9, window_size=W)
        doubled = ParametersNotChanging(rtol=1e-9, window_size=W)
        seq_single = [_params([0.0, 0.0]), _params([0.3, 0.4])]
        seq_doubled = [
            [tf.constant([0.0, 0.0, 0.0, 0.0], tf.float64)],
            [tf.constant([0.3, 0.4, 0.3, 0.4], tf.float64)],
        ]
        s1, _ = _run(single, seq_single)
        s2, _ = _run(doubled, seq_doubled)
        assert s1[0].average_change.numpy() == pytest.approx(
            s2[0].average_change.numpy()
        )

    def test_rel_change_is_ewma_over_param_norm(self):
        W = 10
        c = ParametersNotChanging(rtol=1e-9, window_size=W)
        seq = [_params(10.0), _params(11.0), _params(11.5)]
        states, _ = _run(c, seq)
        for state, val in zip(states, [11.0, 11.5]):
            expected = state.average_change.numpy() / abs(val)
            assert state.rel_change.numpy() == pytest.approx(expected, rel=1e-9)

    def test_ewma_decays_on_static_parameters(self):
        W = 5
        c = ParametersNotChanging(rtol=1e-12, window_size=W)
        seq = [_params(0.0), _params(1.0)] + [_params(1.0)] * 20
        states, _ = _run(c, seq)
        decay = 1.0 - 1.0 / W
        ewma_first = (1.0 - 0.0) / W
        for i, state in enumerate(states[1:], start=1):
            assert state.average_change.numpy() == pytest.approx(
                ewma_first * decay**i, rel=1e-9
            )


class TestConvergence:
    def test_rtol_fires_when_parameters_settle(self):
        c = ParametersNotChanging(
            rtol=1e-3, min_num_steps=0, min_consecutive=1, window_size=2
        )
        # One jump, then perfectly static
        seq = [_params(0.0), _params(5.0)] + [_params(5.0)] * 50
        _, converged_at = _run(c, seq)
        assert converged_at is not None

    def test_does_not_fire_while_changing(self):
        c = ParametersNotChanging(
            rtol=1e-6, min_num_steps=0, min_consecutive=1, window_size=2
        )
        # Parameters keep moving by a large relative amount every step
        seq = [_params(float(i)) for i in range(1, 30)]
        _, converged_at = _run(c, seq)
        assert converged_at is None

    def test_atol_fires(self):
        c = ParametersNotChanging(
            atol=0.1, min_num_steps=0, min_consecutive=1, window_size=2
        )
        seq = [_params(0.0), _params(1.0)] + [_params(1.0)] * 50
        _, converged_at = _run(c, seq)
        assert converged_at is not None

    def test_respects_min_num_steps(self):
        c = ParametersNotChanging(
            rtol=1e6, min_num_steps=50, min_consecutive=1, window_size=2
        )
        seq = [_params(1.0)] * 10
        _, converged_at = _run(c, seq)
        assert converged_at is None

    def test_fires_after_min_consecutive(self):
        c = ParametersNotChanging(
            rtol=1e6, min_num_steps=0, min_consecutive=4, window_size=2
        )
        seq = [_params(1.0)] * 20
        _, converged_at = _run(c, seq)
        assert converged_at == 4

    def test_must_specify_rtol_or_atol(self):
        with pytest.raises(ValueError, match="rtol"):
            ParametersNotChanging()


class TestNaNHandling:
    def test_nonfinite_step_freezes_previous_parameters(self):
        c = ParametersNotChanging(rtol=1e-4, window_size=10)
        seq = [_params(1.0), _params(float("nan")), _params(1.0)]
        states, _ = _run(c, seq)
        # After the NaN step previous_parameters should still be 1.0, so the
        # change on the final step is measured as 0 (finite).
        assert np.isfinite(states[0].average_change.numpy())  # the NaN step
        assert np.isfinite(states[1].average_change.numpy())  # recovery step

    def test_convergence_not_declared_on_nonfinite_step(self):
        c = ParametersNotChanging(
            rtol=1e9, min_num_steps=0, min_consecutive=1, window_size=2
        )
        seq = [_params(1.0), _params(float("nan"))]
        _, converged_at = _run(c, seq)
        assert converged_at is None

    def test_nonfinite_resets_consecutive_counter(self):
        c = ParametersNotChanging(
            rtol=1e9, min_num_steps=0, min_consecutive=3, window_size=2
        )
        seq = [_params(1.0), _params(1.0), _params(1.0), _params(float("nan"))]
        states, _ = _run(c, seq)
        assert states[-1].consecutive_below.numpy() == 0
