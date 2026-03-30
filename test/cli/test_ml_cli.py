import pytest
from allpairspy import AllPairs
from click.testing import CliRunner
from treeflow.cli.ml import treeflow_ml

pytestmark = pytest.mark.cli

_DATASETS = [
    ("hello.nwk", "hello.fasta", False),
    ("wnv.nwk", "wnv.fasta", True),
]
_MODEL_FILES = [None, "model.yaml"]
_INIT = [True, False]

_CASES = list(AllPairs([_DATASETS, _MODEL_FILES, _INIT]))


def _case_id(case):
    dataset, model_filename, include_init_values = case
    return "-".join([
        dataset[0].removesuffix(".nwk"),
        model_filename.removesuffix(".yaml") if model_filename else "no-model",
        "init" if include_init_values else "no-init",
    ])


@pytest.mark.parametrize(
    "dataset,model_filename,include_init_values",
    _CASES,
    ids=[_case_id(c) for c in _CASES],
)
def test_ml_cli(
    test_data_dir,
    samples_output_path,
    tree_samples_output_path,
    trace_output_path,
    dataset,
    model_filename,
    include_init_values,
):
    import pandas as pd
    import dendropy

    newick_filename, fasta_filename, _ = dataset
    newick_file = str(test_data_dir / newick_filename)
    fasta_file = str(test_data_dir / fasta_filename)
    model_file = str(test_data_dir / model_filename) if model_filename is not None else None

    runner = CliRunner()
    args = [
        "-i", fasta_file,
        "-t", newick_file,
        "-n", str(10),
        "--variables-output", str(samples_output_path),
        "--tree-output", str(tree_samples_output_path),
        "--trace-output", str(trace_output_path),
    ]
    if model_file is None:
        init_values_string = "clock_rate=0.01"
    else:
        args += ["-m", model_file]
        init_values_string = "pop_size=10"

    if include_init_values:
        args += ["--init-values", init_values_string]

    res = runner.invoke(treeflow_ml, args, catch_exceptions=False)
    assert res.exit_code == 0
    print(res.stdout)
    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == 1

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == 1
