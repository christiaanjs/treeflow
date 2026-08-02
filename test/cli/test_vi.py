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
        "run",
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


def test_vi_resume_from_trace(test_data_dir, trace_output_path, tmp_path):
    import pickle
    import numpy as np

    newick_file = str(test_data_dir / "hello.nwk")
    fasta_file = str(test_data_dir / "hello.fasta")
    resumed_trace_path = tmp_path / "resumed-trace.pickle"

    runner = CliRunner()
    first_res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "mean_field",
            "--trace-output", str(trace_output_path),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert first_res.exit_code == 0

    resumed_res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "mean_field",
            "--resume-from-trace", str(trace_output_path),
            "--trace-output", str(resumed_trace_path),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert resumed_res.exit_code == 0
    assert "Resuming from trace" in resumed_res.stdout

    with open(trace_output_path, "rb") as f:
        first_trace = pickle.load(f)
    with open(resumed_trace_path, "rb") as f:
        resumed_trace = pickle.load(f)

    fresh_res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "mean_field",
            "-s", "1",
            "--trace-output", str(tmp_path / "fresh-trace.pickle"),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert fresh_res.exit_code == 0
    with open(tmp_path / "fresh-trace.pickle", "rb") as f:
        fresh_trace = pickle.load(f)

    # The resumed run should start (much) closer to where the first run left off
    # than a freshly-initialised run does.
    for name, first_final in first_trace.parameters.items():
        resumed_start = np.asarray(resumed_trace.parameters[name])[0]
        fresh_start = np.asarray(fresh_trace.parameters[name])[0]
        first_final = np.asarray(first_final)[-1]
        resumed_dist = np.linalg.norm(resumed_start - first_final)
        fresh_dist = np.linalg.norm(fresh_start - first_final)
        assert resumed_dist < fresh_dist


def test_vi_plot(test_data_dir, trace_output_path, tmp_path):
    newick_file = str(test_data_dir / "hello.nwk")
    fasta_file = str(test_data_dir / "hello.fasta")

    runner = CliRunner()
    run_res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "mean_field",
            "--trace-output", str(trace_output_path),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0

    full_plot_path = tmp_path / "full.png"
    full_res = runner.invoke(
        treeflow_vi,
        ["plot", "-t", str(trace_output_path), "-o", str(full_plot_path)],
        catch_exceptions=False,
    )
    assert full_res.exit_code == 0
    assert full_plot_path.exists()

    sample_plot_path = tmp_path / "sample.png"
    sample_res = runner.invoke(
        treeflow_vi,
        [
            "plot",
            "-t", str(trace_output_path),
            "-o", str(sample_plot_path),
            "--sample",
            "--title", "test run",
        ],
        catch_exceptions=False,
    )
    assert sample_res.exit_code == 0
    assert sample_plot_path.exists()


def test_vi_max_trace_coords(test_data_dir, trace_output_path):
    import pickle
    import numpy as np

    newick_file = str(test_data_dir / "hello.nwk")
    fasta_file = str(test_data_dir / "hello.fasta")

    runner = CliRunner()
    max_trace_coords = 3
    res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "full_rank",
            "--max-trace-coords", str(max_trace_coords),
            "--trace-output", str(trace_output_path),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0

    with open(trace_output_path, "rb") as f:
        trace = pickle.load(f)

    # Every traced variable's per-step coordinate count is capped at
    # max_trace_coords (small variables below the cap are still traced in full).
    saw_capped_variable = False
    for value in trace.parameters.values():
        arr = np.asarray(value)
        n_coords = int(np.prod(arr.shape[1:]))
        assert n_coords <= max_trace_coords
        if n_coords == max_trace_coords:
            saw_capped_variable = True
    # full_rank's D x D scale matrix should be large enough to actually get
    # capped for this test to be meaningful.
    assert saw_capped_variable

    # The trace records which coordinates it kept, so plots can label them by
    # coordinate rather than by position in the trace.
    assert set(trace.parameter_coords) == set(trace.parameters)
    for name, coords in trace.parameter_coords.items():
        n_coords = int(np.prod(np.asarray(trace.parameters[name]).shape[1:]))
        assert len(coords.indices) == n_coords
        assert list(coords.indices) == sorted(set(coords.indices))
        # The last coordinate -- the root, for a node-height vector -- is
        # always kept.
        assert coords.indices[-1] == coords.size - 1


def test_vi_plot_subsampled_trace(test_data_dir, trace_output_path, tmp_path):
    """`plot` works on a trace written with `run --max-trace-coords`."""
    newick_file = str(test_data_dir / "hello.nwk")
    fasta_file = str(test_data_dir / "hello.fasta")

    runner = CliRunner()
    run_res = runner.invoke(
        treeflow_vi,
        [
            "run",
            "-i", fasta_file,
            "-t", newick_file,
            "-n", "5",
            "-va", "mean_field",
            "--max-trace-coords", "2",
            "--trace-output", str(trace_output_path),
            "--no-progress-bar",
        ],
        catch_exceptions=False,
    )
    assert run_res.exit_code == 0

    for layout, filename in [("--full", "full.png"), ("--sample", "sample.png")]:
        plot_path = tmp_path / filename
        res = runner.invoke(
            treeflow_vi,
            ["plot", "-t", str(trace_output_path), "-o", str(plot_path), layout],
            catch_exceptions=False,
        )
        assert res.exit_code == 0
        assert plot_path.exists()
