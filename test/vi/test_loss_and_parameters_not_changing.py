import numpy as np
import tensorflow as tf
from tensorflow_probability.python.math.minimize import minimize
from treeflow.vi.convergence_criteria.loss_and_parameters_not_changing import (
    LossAndParametersNotChanging,
)


class TestConstruction:
    def test_shared_rtol_propagates_to_both(self):
        c = LossAndParametersNotChanging(rtol=1e-5)
        assert c.loss_criterion.rtol == 1e-5
        assert c.parameters_criterion.rtol == 1e-5

    def test_param_atol_only_on_parameters(self):
        c = LossAndParametersNotChanging(rtol=1e-5, param_atol=1e-3)
        assert c.parameters_criterion.atol == 1e-3
        assert c.loss_criterion.atol is None

    def test_shared_settings_propagate(self):
        c = LossAndParametersNotChanging(
            rtol=1e-5, window_size=42, min_num_steps=7, min_consecutive=3
        )
        for sub in (c.loss_criterion, c.parameters_criterion):
            assert sub.window_size == 42
            assert sub.min_num_steps.numpy() == 7
            assert sub.min_consecutive == 3

    def test_default_mode_is_all(self):
        assert LossAndParametersNotChanging(rtol=1e-5).mode == "all"


class TestIntegration:
    def _fit(self, criterion, num_steps=4000):
        x = tf.Variable(5.0, dtype=tf.float64)
        opt = tf.keras.optimizers.Adam(0.1)
        _, converged = minimize(
            lambda: (x - 3.0) ** 2 + 1.0,
            num_steps=num_steps,
            optimizer=opt,
            convergence_criterion=criterion,
            trace_fn=lambda t: (t.loss, t.has_converged),
        )
        c = converged.numpy()
        first = int(np.argmax(c)) if c.any() else None
        return x.numpy(), first

    def test_converges_on_nonzero_optimum(self):
        x, first = self._fit(
            LossAndParametersNotChanging(
                rtol=1e-5, min_num_steps=0, min_consecutive=2, window_size=10
            )
        )
        assert first is not None
        assert x == np.float64(x)  # finite
        assert abs(x - 3.0) < 1e-2

    def test_param_atol_enables_zero_optimum_convergence(self):
        """With optimum at 0, rtol alone stalls; param_atol lets it converge."""
        x = tf.Variable(5.0, dtype=tf.float64)
        opt = tf.keras.optimizers.Adam(0.1)

        def loss_fn():
            return x**2

        no_atol = LossAndParametersNotChanging(
            rtol=1e-5, min_num_steps=0, min_consecutive=2, window_size=10
        )
        converged = minimize(
            loss_fn, num_steps=4000, optimizer=opt,
            convergence_criterion=no_atol,
            trace_fn=lambda t: t.has_converged,
        )
        assert not converged.numpy().any()  # rtol cannot settle near zero

        # The parameter term on its own, with an absolute floor, fires.
        x.assign(5.0)
        opt2 = tf.keras.optimizers.Adam(0.1)
        with_atol = LossAndParametersNotChanging(
            rtol=1e-5, param_atol=1e-4, mode="any",
            min_num_steps=0, min_consecutive=2, window_size=10,
        )
        converged2 = minimize(
            loss_fn, num_steps=4000, optimizer=opt2,
            convergence_criterion=with_atol,
            trace_fn=lambda t: t.has_converged,
        )
        assert converged2.numpy().any()
