import pytest
from treeflow.cli.vi import treeflow_vi
from click.testing import CliRunner


@pytest.fixture
def approx_output_path(tmp_path):
    return tmp_path / "approx-output.pickle"


@pytest.mark.parametrize("include_init_values", [True, False])
def test_vi(newick_fasta_file_dated, include_init_values):
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
    if include_init_values:
        args = args + [
            "--init-values",
            "rate=0.01",
        ]
    res = runner.invoke(
        treeflow_vi,
        args,
        catch_exceptions=False,
    )
    print(res.stdout)
