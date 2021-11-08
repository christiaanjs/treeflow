import os
import pathlib
import pytest
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from functools import partial


@pytest.fixture
def tensor_constant():
    return partial(tf.constant, dtype=DEFAULT_FLOAT_DTYPE_TF)


pytest_plugins = [
    "fixtures.tree_fixtures",
    "fixtures.data_fixtures",
    "fixtures.substitution_fixtures",
]

if os.getenv("_PYTEST_RAISE", "0") != "0":
    # Stop pytest catching exceptions in debug run configuration
    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture
def test_data_dir():
    return pathlib.Path("test") / "data"
