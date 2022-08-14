import pytest
import tensorflow as tf
import numpy as np
import treeflow
from treeflow.vi.optimizers.robust_optimizer import RobustOptimizer
from numpy.testing import assert_allclose
import tensorflow_probability as tfp


@pytest.fixture
def base_optimizer_factory():
    return lambda: tf.optimizers.SGD(learning_rate=1e-2)


@pytest.fixture
def base_optimizer(base_optimizer_factory):
    return base_optimizer_factory()


@pytest.fixture
def robust_optimizer(base_optimizer_factory):
    return RobustOptimizer(base_optimizer_factory(), max_retries=10)


@pytest.fixture
def vars_builder():
    def vars(prefix=""):
        return dict(
            a=tf.Variable(
                tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
                name=f"{prefix}a",
            ),
            b=tf.Variable(
                tf.convert_to_tensor([1.2, 3.2], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
                name=f"{prefix}b",
            ),
        )

    return vars


@pytest.fixture
def grads():
    return dict(
        a=tf.convert_to_tensor([-1.0, 1.2, 0.3], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
        b=tf.convert_to_tensor(
            [[0.2, 2.1], [-0.1, 0.5], [0.1, 2.2]], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
        ),
    )


@pytest.fixture
def grads_with_nan(grads):
    return dict(
        a=tf.convert_to_tensor(
            [-1.0, 2.3, -0.2, 1.2, np.nan, 0.3], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
        ),
        b=tf.convert_to_tensor(
            [
                [0.2, 2.1],
                [np.nan, 0.2],
                [-0.1, np.nan],
                [-0.1, 0.5],
                [1.1, 2.1],
                [0.1, 2.2],
            ],
            dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF,
        ),
    )


@pytest.fixture
def vars(vars_builder):
    return vars_builder()


@pytest.fixture
def other_vars(vars_builder):
    return vars_builder("robust_")


def test_robust_optimizer_matches(
    base_optimizer: tf.optimizers.Optimizer,
    robust_optimizer: RobustOptimizer,
    grads,
    grads_with_nan,
    vars,
    other_vars,
):

    keys = list(vars.keys())

    vars_list = [vars[key] for key in keys]
    grads_list = [grads[key] for key in keys]
    for i in range(grads[keys[0]].shape[0]):
        base_optimizer.apply_gradients(zip([grad[i] for grad in grads_list], vars_list))

    other_vars_list = [other_vars[key] for key in keys]
    grads_with_nan_list = [grads_with_nan[key] for key in keys]
    for i in range(grads_with_nan[keys[0]].shape[0]):
        robust_optimizer.apply_gradients(
            zip([grad[i] for grad in grads_with_nan_list], other_vars_list)
        )

    for key in keys:
        assert np.all(np.isfinite(other_vars[key]))
        assert_allclose(vars[key].numpy(), other_vars[key].numpy())


# def test_robust_optimizer_throws():
#     assert False


def obj(vars):
    return lambda: tf.math.square(vars["a"]) + tf.math.reduce_sum(
        tf.math.square(vars["a"] - vars["b"])
    )


def test_robust_optimizer_matches_with_minimize(
    base_optimizer: tf.optimizers.Optimizer,
    robust_optimizer: RobustOptimizer,
    vars,
    other_vars,
):
    obj_applied = obj(vars)
    other_obj_applied = obj(other_vars)

    init = {key: var.numpy() for key, var in vars.items()}

    num_steps = 10
    res = tfp.math.minimize(obj_applied, num_steps, base_optimizer)
    other_res = tfp.math.minimize(other_obj_applied, num_steps, robust_optimizer)

    for key in vars.keys():
        assert not np.allclose(vars[key].numpy(), init[key])
        assert_allclose(vars[key].numpy(), other_vars[key].numpy())


NOISY_STEP = 3


def noisy_obj(vars):
    count_var = tf.Variable(0)
    inner_obj = obj(vars)

    def _noisy_obj():
        count_val = tf.identity(count_var)
        count_var.assign_add(1)
        return tf.cond(count_val == NOISY_STEP, lambda: inner_obj() * np.nan, inner_obj)

    return _noisy_obj


def test_robust_optimizer_minimize_with_nans(robust_optimizer: RobustOptimizer, vars):
    noisy_obj_applied = noisy_obj(vars)
    res = tfp.math.minimize(
        noisy_obj_applied,
        NOISY_STEP + 3,
        robust_optimizer,
        trainable_variables=list(vars.values()),
    )
    assert not np.isfinite(res[NOISY_STEP])
    for key in vars.keys():
        assert np.all(np.isfinite(vars[key].numpy()))


@pytest.fixture
def grads_with_too_many_nans():
    return dict(
        a=tf.convert_to_tensor(
            [-1.0] + ([np.nan] * 9) + [2.3, 0.3], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
        ),
        b=tf.convert_to_tensor(
            [[0.2, 2.1], [-0.1, 0.2]]
            + [[-0.1, np.nan], [np.nan, 0.5]] * 9
            + [
                [1.1, 2.1],
                [0.1, 2.2],
            ],
            dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF,
        ),
    )


def test_robust_optimizer_doesnt_throw(base_optimizer, grads_with_too_many_nans, vars):
    robust_optimizer = RobustOptimizer(base_optimizer, max_retries=11)
    keys = list(vars.keys())

    vars_list = [vars[key] for key in keys]
    grads_list = [grads_with_too_many_nans[key] for key in keys]
    for i in range(grads_list[0].shape[0]):
        robust_optimizer.apply_gradients(
            zip([grad[i] for grad in grads_list], vars_list)
        )


def test_robust_optimizer_throws(
    robust_optimizer: RobustOptimizer, grads_with_too_many_nans, vars
):
    keys = list(vars.keys())

    vars_list = [vars[key] for key in keys]
    grads_list = [grads_with_too_many_nans[key] for key in keys]
    with pytest.raises(Exception) as ex:
        for i in range(grads_list[0].shape[0]):
            robust_optimizer.apply_gradients(
                zip([grad[i] for grad in grads_list], vars_list)
            )


def really_noisy_obj(vars):
    count_var = tf.Variable(0)
    inner_obj = obj(vars)

    def _noisy_obj():
        count_val = tf.identity(count_var)
        count_var.assign_add(1)
        return tf.cond(count_val > NOISY_STEP, lambda: inner_obj() * np.nan, inner_obj)

    return _noisy_obj


def test_robust_optimizer_throws_with_minimize(robust_optimizer: RobustOptimizer, vars):
    noisy_obj_applied = really_noisy_obj(vars)
    with pytest.raises(Exception) as ex:
        res = tfp.math.minimize(
            noisy_obj_applied,
            NOISY_STEP + robust_optimizer.max_retries + 2,
            robust_optimizer,
            trainable_variables=list(vars.values()),
        )
        print(res)
