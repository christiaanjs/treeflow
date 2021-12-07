import pytest
from treeflow.cli.benchmark import treeflow_benchmark
from click.testing import CliRunner


@pytest.fixture
def benchmark_output_path(tmp_path):
    return tmp_path / "treeflow-benchmark.csv"


def test_benchmark(hello_fasta_file, hello_newick_file, benchmark_output_path):
    runner = CliRunner()
    res = runner.invoke(
        treeflow_benchmark,
        [
            "-i",
            hello_fasta_file,
            "-t",
            hello_newick_file,
            "-r",
            str(1),
            "-o",
            str(benchmark_output_path),
        ],
        catch_exceptions=False,
    )
    print(res)
