import typing as tp
from typing_extensions import Protocol
from functools import partial
import tqdm
import tensorflow as tf
from tensorflow_probability.python.math.minimize import MinimizeTraceableQuantities


class ProgressBarFunc(Protocol):
    def __call__(self, total: int, *args, **kwds) -> tqdm.tqdm:
        ...


class ProgressBarTraceFunctionContextManager:
    def __init__(self, tqdm: tp.Optional[tqdm.tqdm], trace_fn: tp.Callable):
        self.tqdm = tqdm
        self.trace_fn = trace_fn

    def __enter__(self):
        if self.tqdm is not None:
            self.tqdm.__enter__()
        return self.trace_fn

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tqdm is not None:
            self.tqdm.__exit__(exc_type, exc_value, traceback)


def update_trace_fn(
    mtq: MinimizeTraceableQuantities,
    trace_fn: tp.Callable,
    tqdm_instance: tqdm.tqdm,
    update_step: int = 10,
):
    step = mtq.step
    if (step % update_step == 0 and tqdm_instance.n < step) or tqdm_instance.n == step:
        tqdm_instance.update(update_step)
    return trace_fn(mtq)


def make_progress_bar_trace_fn(
    trace_fn: tp.Callable,
    num_steps: int,
    progress_bar: tp.Union[ProgressBarFunc, bool] = True,
    update_step: int = 10,
):

    total = num_steps
    if isinstance(progress_bar, bool):
        if progress_bar:
            tqdm_instance = tqdm.tqdm(total=total)
        else:
            tqdm_instance = None
    else:
        tqdm_instance = progress_bar(total=total)

    if tqdm_instance is None:
        wrapped_trace_fn = trace_fn
    else:
        wrapped_trace_fn = partial(
            update_trace_fn,
            trace_fn=trace_fn,
            tqdm_instance=tqdm_instance,
            update_step=update_step,
        )

    return ProgressBarTraceFunctionContextManager(tqdm_instance, wrapped_trace_fn)


def _default_hmc_trace_fn(current_state, kernel_results):
    return kernel_results


def make_hmc_progress_bar_trace_fn(
    trace_fn: tp.Callable,
    num_results: int,
    num_burnin_steps: int,
    progress_bar: tp.Union[ProgressBarFunc, bool] = True,
    update_step: int = 10,
):
    total = num_results + num_burnin_steps

    if isinstance(progress_bar, bool):
        if progress_bar:
            tqdm_instance = tqdm.tqdm(total=total)
        else:
            tqdm_instance = None
    else:
        tqdm_instance = progress_bar(total=total)

    if tqdm_instance is None:
        return ProgressBarTraceFunctionContextManager(None, trace_fn)

    step_var = tf.Variable(0, trainable=False, dtype=tf.int32)

    def wrapped_trace_fn(current_state, kernel_results):
        step_var.assign_add(1)

        def _update(step):
            s = int(step)
            if (s % update_step == 0 and tqdm_instance.n < s) or s == total:
                tqdm_instance.update(min(update_step, total - tqdm_instance.n))
            return s

        tf.py_function(_update, [step_var], tf.int32)
        return trace_fn(current_state, kernel_results)

    return ProgressBarTraceFunctionContextManager(tqdm_instance, wrapped_trace_fn)
