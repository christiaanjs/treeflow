import pytest
from allpairspy import AllPairs
from treeflow.cli.vi import treeflow_vi, approximation_builders
from click.testing import CliRunner

pytestmark = pytest.mark.cli

_HELLO = ("hello.nwk", "hello.fasta", False)
_WNV = ("wnv.nwk", "wnv.fasta", True)
_MODEL_FILES = [None, "model.yaml", "yule-model.yaml"]
_APPROX = list(approximation_builders.keys())
_INIT = [True, False]
_CONVERGENCE = [None, "nonfinite"]

_INIT_VALUES = {
    None: "clock_rate=0.01",
    "model.yaml": "pop_size=10",
    "yule-model.yaml": "birth_rate=2,frequencies=0.24|0.23|0.26|0.27",
}

# VI is dominated by TF graph tracing and is dataset-insensitive in practice
# (a wnv run costs about the same as hello), so we pairwise-cover the option
# surface -- model file, approximation, init values, convergence criterion -- on
# the tiny hello dataset, exercise the progress bar in a single case, and add one
# wnv smoke to cover the large-alignment / many-branch path end to end.
_CASES = []
for _i, (_model, _approx, _init, _conv) in enumerate(
    AllPairs([_MODEL_FILES, _APPROX, _INIT, _CONVERGENCE])
):
    _CASES.append((_HELLO, _model, _approx, _i == 0, _init, _conv))
_CASES.append((_WNV, None, "mean_field", False, False, None))


def _case_id(case):
    dataset, model_filename, approx, progress_bar, include_init_values, convergence = case
    return "-".join([
        dataset[0].removesuffix(".nwk"),
        model_filename.removesuffix(".yaml") if model_filename else "no-model",
        approx,
        "progress" if progress_bar else "no-progress",
        "init" if include_init_values else "no-init",
        convergence if convergence else "default-convergence",
    ])


@pytest.mark.parametrize(
    "dataset,model_filename,variational_approximation,progress_bar,include_init_values,convergence_criterion",
    _CASES,
    ids=[_case_id(c) for c in _CASES],
)
def test_vi(
    test_data_dir,
    samples_output_path,
    tree_samples_output_path,
    dataset,
    model_filename,
    variational_approximation,
    progress_bar,
    include_init_values,
    convergence_criterion,
):
    import pandas as pd
    import dendropy

    newick_filename, fasta_filename, _ = dataset
    newick_file = str(test_data_dir / newick_filename)
    fasta_file = str(test_data_dir / fasta_filename)
    model_file = str(test_data_dir / model_filename) if model_filename is not None else None

    runner = CliRunner()
    n_output_samples = 10
    args = [
        "-i", fasta_file,
        "-t", newick_file,
        "-n", str(10),
        "-va", variational_approximation,
        "--samples-output", str(samples_output_path),
        "--tree-samples-output", str(tree_samples_output_path),
        "--n-output-samples", str(n_output_samples),
        "--progress-bar" if progress_bar else "--no-progress-bar",
    ]
    if model_file is not None:
        args += ["-m", model_file]
    if include_init_values:
        args += ["--init-values", _INIT_VALUES[model_filename]]
    if convergence_criterion is not None:
        args += ["--convergence-criterion", convergence_criterion]

    res = runner.invoke(treeflow_vi, args, catch_exceptions=False)
    assert res.exit_code == 0
    print(res.stdout)
    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == n_output_samples

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == n_output_samples
