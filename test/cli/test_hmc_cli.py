import pytest
from allpairspy import AllPairs
from treeflow.cli.hmc import treeflow_hmc, KERNEL_CHOICES
from click.testing import CliRunner

pytestmark = pytest.mark.cli

_DATASETS = [
    ("hello.nwk", "hello.fasta"),
    ("wnv.nwk", "wnv.fasta"),
]
_MODEL_FILES = [None, "model.yaml"]
_KERNELS = KERNEL_CHOICES
_INIT = [True, False]

_INIT_VALUES = {
    None: "clock_rate=0.01",
    "model.yaml": "pop_size=10",
}

_CASES = list(AllPairs([_DATASETS, _MODEL_FILES, _KERNELS, _INIT]))


def _case_id(case):
    dataset, model_filename, kernel, include_init_values = case
    return "-".join([
        dataset[0].removesuffix(".nwk"),
        model_filename.removesuffix(".yaml") if model_filename else "no-model",
        kernel,
        "init" if include_init_values else "no-init",
    ])


@pytest.mark.parametrize(
    "dataset,model_filename,kernel,include_init_values",
    _CASES,
    ids=[_case_id(c) for c in _CASES],
)
def test_hmc(
    test_data_dir,
    samples_output_path,
    tree_samples_output_path,
    dataset,
    model_filename,
    kernel,
    include_init_values,
):
    import pandas as pd
    import dendropy

    newick_filename, fasta_filename = dataset
    newick_file = str(test_data_dir / newick_filename)
    fasta_file = str(test_data_dir / fasta_filename)
    model_file = str(test_data_dir / model_filename) if model_filename is not None else None

    runner = CliRunner()
    num_results = 10
    args = [
        "-i", fasta_file,
        "-t", newick_file,
        "-n", str(num_results),
        "--num-burnin-steps", str(3),
        "--num-adaptation-steps", str(3),
        "--num-leapfrog-steps", str(3),
        "--kernel", kernel,
        "--samples-output", str(samples_output_path),
        "--tree-samples-output", str(tree_samples_output_path),
    ]
    if model_file is not None:
        args += ["-m", model_file]
    if include_init_values:
        args += ["--init-values", _INIT_VALUES[model_filename]]

    res = runner.invoke(treeflow_hmc, args, catch_exceptions=False)
    assert res.exit_code == 0, res.stdout
    print(res.stdout)

    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == num_results

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == num_results
