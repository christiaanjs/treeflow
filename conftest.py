import os
import sys
import pathlib
import pytest
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from functools import partial


conftest_dir = pathlib.Path(__file__).parents[0]
sys.path.append(str(conftest_dir / "test" / "helpers"))
sys.path.append(str(conftest_dir / "test" / "fixtures"))


@pytest.fixture
def tensor_constant():
    return partial(tf.constant, dtype=DEFAULT_FLOAT_DTYPE_TF)


pytest_plugins = [
    "tree_fixtures",
    "data_fixtures",
    "ratio_fixtures",
    "substitution_fixtures",
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
