import pytest
from treeflow.cli.vi import treeflow_vi
from click.testing import CliRunner


@pytest.fixture
def approx_output_path(tmp_path):
    return tmp_path / "approx-output.pickle"


def test_vi(newick_fasta_file_dated, approx_output_path):
    newick_file, fasta_file, dated = newick_fasta_file_dated
    runner = CliRunner()
    res = runner.invoke(
        treeflow_vi,
        [
            "-i",
            str(fasta_file),
            "-t",
            str(newick_file),
            "-n",
            str(10),
            "--init-values",
            "rate=0.01",
        ],
        catch_exceptions=False,
    )
    print(res.stdout)
