import pytest
from treeflow.cli.vi import treeflow_vi
from click.testing import CliRunner


@pytest.fixture
def approx_output_path(tmp_path):
    return tmp_path / "approx-output.pickle"


@pytest.fixture(params=[None, "model.yaml"])
def model_file(request, test_data_dir):
    filename = request.param
    if filename is None:
        return None
    else:
        return str(test_data_dir / filename)


@pytest.mark.parametrize("include_init_values", [True, False])
def test_vi(newick_fasta_file_dated, include_init_values, model_file):
    newick_file, fasta_file, dated = newick_fasta_file_dated
    runner = CliRunner()
    args = [
        "-i",
        str(fasta_file),
        "-t",
        str(newick_file),
        "-n",
        str(10),
    ]
    if model_file is None:
        init_values_string = "rate=0.01"
    else:
        args = args + ["-m", model_file]
        init_values_string = "pop_size=10"

    if include_init_values:
        args = args + ["--init-values", init_values_string]
    res = runner.invoke(
        treeflow_vi,
        args,
        catch_exceptions=False,
    )
    print(res.stdout)
