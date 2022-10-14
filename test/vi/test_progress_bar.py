import typing as tp
import pytest
import tensorflow as tf
import tqdm
from tensorflow_probability.python.math.minimize import minimize
from treeflow.vi.progress_bar import make_progress_bar_trace_fn


@pytest.mark.parametrize("update_step", [1, 3])
def test_make_progress_bar_trace_fn(update_step):
    x = tf.Variable(0.0)
    loss = lambda: tf.square(x) + 2 * x - 2

    trace_fn = lambda mtq: mtq.loss
    num_steps = 12

    tqdm_instance: tp.Optional[tqdm.tqdm] = None

    def make_tqdm(total):
        nonlocal tqdm_instance
        tqdm_instance = tqdm.tqdm(total=total)
        return tqdm_instance

    with make_progress_bar_trace_fn(
        trace_fn, num_steps, make_tqdm, update_step=update_step
    ) as progress_trace_fn:
        trace = minimize(
            loss, num_steps, tf.optimizers.Adam(), trace_fn=progress_trace_fn
        )

    assert tqdm_instance is not None
    assert tqdm_instance.n == tqdm_instance.total
    assert tqdm_instance.disable
