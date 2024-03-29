from operator import gt
import pytest
from treeflow.cli.benchmark import treeflow_benchmark
from click.testing import CliRunner


@pytest.fixture
def benchmark_output_path(tmp_path):
    return tmp_path / "treeflow-benchmark.csv"


@pytest.mark.parametrize(["use_bito", "gtr"], [(True, False), (False, True)])
def test_benchmark(
    hello_fasta_file, hello_newick_file, benchmark_output_path, use_bito, gtr
):
    runner = CliRunner()
    args = [
        "-i",
        hello_fasta_file,
        "-t",
        hello_newick_file,
        "-r",
        str(1),
        "-o",
        str(benchmark_output_path),
    ]
    if use_bito:
        args.append("--use-bito")
    if gtr:
        args.append("--gtr")
    res = runner.invoke(
        treeflow_benchmark,
        args,
        catch_exceptions=False,
    )
    print(res)
