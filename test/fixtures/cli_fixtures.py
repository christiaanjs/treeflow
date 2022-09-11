import pytest


@pytest.fixture
def trace_output_path(tmp_path):
    return tmp_path / "trace.pickle"


@pytest.fixture
def samples_output_path(tmp_path):
    return tmp_path / "approx-samples.csv"


@pytest.fixture
def tree_samples_output_path(tmp_path):
    return tmp_path / "approx-tree-samples.nexus"


@pytest.fixture
def actual_model_file(test_data_dir):
    return str(test_data_dir / "model.yaml")


@pytest.fixture(params=[None, "model.yaml"])
def model_file(request, test_data_dir):
    filename = request.param
    if filename is None:
        return None
    else:
        return str(test_data_dir / filename)
