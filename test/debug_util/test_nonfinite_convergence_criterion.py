from tensorflow_probability.python.math.minimize import minimize as tfp_minimize
from treeflow_test_helpers.optimization_helpers import obj, vars, optimizer_builder
from treeflow.debug.nonfinite_convergence_criterion import NonfiniteConvergenceCriterion

# TODO: More comprehensive tests


def test_NonfiniteConvergenceCriterion_runs():
    optimizer = optimizer_builder()
    convergence_criterion = NonfiniteConvergenceCriterion()

    _vars = vars()
    _obj = obj(_vars)

    res = tfp_minimize(
        _obj,
        10,
        optimizer,
        return_full_length_trace=False,
        convergence_criterion=convergence_criterion,
    )
