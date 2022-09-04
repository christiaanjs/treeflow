import pytest
from click.testing import CliRunner
from treeflow.cli.ml import treeflow_ml


@pytest.mark.parametrize("include_init_values", [True, False])
def test_ml_cli(
    newick_fasta_file_dated,
    include_init_values,
    model_file,
    samples_output_path,
    tree_samples_output_path,
    trace_output_path,
):
    import pandas as pd
    import dendropy

    newick_file, fasta_file, dated = newick_fasta_file_dated
    runner = CliRunner()
    args = [
        "-i",
        str(fasta_file),
        "-t",
        str(newick_file),
        "-n",
        str(10),
        "--variables-output",
        str(samples_output_path),
        "--tree-output",
        str(tree_samples_output_path),
        "--trace-output",
        str(trace_output_path),
    ]
    if model_file is None:
        init_values_string = "rate=0.01"
    else:
        args = args + ["-m", model_file]
        init_values_string = "pop_size=10"

    if include_init_values:
        args = args + ["--init-values", init_values_string]
    res = runner.invoke(
        treeflow_ml,
        args,
        catch_exceptions=False,
    )
    print(res.stdout)
    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == 1

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == 1
