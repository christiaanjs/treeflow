import pytest
from treeflow.cli.vi import treeflow_vi
from click.testing import CliRunner


@pytest.fixture
def samples_output_path(tmp_path):
    return tmp_path / "approx-samples.csv"


@pytest.fixture
def tree_samples_output_path(tmp_path):
    return tmp_path / "approx-tree-samples.nexus"


@pytest.fixture(params=[None, "model.yaml"])
def model_file(request, test_data_dir):
    filename = request.param
    if filename is None:
        return None
    else:
        return str(test_data_dir / filename)


@pytest.mark.parametrize("include_init_values", [True, False])
def test_vi(
    newick_fasta_file_dated,
    include_init_values,
    model_file,
    samples_output_path,
    tree_samples_output_path,
):
    import pandas as pd
    import dendropy

    newick_file, fasta_file, dated = newick_fasta_file_dated
    runner = CliRunner()
    n_output_samples = 10
    args = [
        "-i",
        str(fasta_file),
        "-t",
        str(newick_file),
        "-n",
        str(10),
        "--samples-output",
        str(samples_output_path),
        "--tree-samples-output",
        str(tree_samples_output_path),
        "--n-output-samples",
        str(n_output_samples),
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
    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == n_output_samples

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == n_output_samples


@pytest.mark.parametrize("include_init_values", [True, False])
def test_vi_yule(
    test_data_dir, hello_newick_file, hello_fasta_file, include_init_values
):
    model_file = str(test_data_dir / "yule-model.yaml")
    init_values_string = "birth_rate=2"
    runner = CliRunner()
    args = [
        "-i",
        str(hello_fasta_file),
        "-t",
        str(hello_newick_file),
        "-n",
        str(10),
        "-m",
        model_file,
    ]
    if include_init_values:
        args = args + ["--init-values", init_values_string]
    res = runner.invoke(
        treeflow_vi,
        args,
        catch_exceptions=False,
    )
    print(res.stdout)
