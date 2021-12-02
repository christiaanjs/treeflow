from treeflow.cli.benchmark import treeflow_benchmark
from click.testing import CliRunner


def test_benchmark(hello_fasta_file, hello_newick_file):
    runner = CliRunner()
    res = runner.invoke(
        treeflow_benchmark,
        ["-i", hello_fasta_file, "-t", hello_newick_file, "-r", str(3)],
        catch_exceptions=False,
    )
    print(res)
