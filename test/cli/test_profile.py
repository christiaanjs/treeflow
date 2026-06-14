import pytest
from click.testing import CliRunner
from treeflow.cli.profile import treeflow_profile

pytestmark = pytest.mark.cli

# The profiler is a developer/CI performance tool; this test only checks that it
# runs end to end and emits a usable CSV on the smallest possible configuration.
# Engine selection downgrades to pure TensorFlow automatically when the native op
# is unavailable, so it is safe to request both here.


def test_profile_runs_and_writes_csv(tmp_path):
    import pandas as pd

    output = tmp_path / "profile.csv"
    runner = CliRunner()
    res = runner.invoke(
        treeflow_profile,
        [
            "-T",
            "8,12",
            "--sites",
            "20",
            "-r",
            "2",
            "-o",
            str(output),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    df = pd.read_csv(output)
    # Two synthetic datasets profiled.
    assert set(df["taxa"]) == {8, 12}
    # The density components plus the full target and the ratio transform.
    assert {"likelihood", "prior", "overhead", "full", "ratio_transform"} <= set(
        df["component"]
    )

    # Within an engine, likelihood + prior + overhead should reconstruct full.
    for (taxa, engine), group in df.groupby(["taxa", "engine"]):
        if engine == "shared":
            continue
        times = group.set_index("component")["time_ms"]
        if {"likelihood", "prior", "overhead", "full"} <= set(times.index):
            parts = times[["likelihood", "prior", "overhead"]].sum()
            assert parts == pytest.approx(times["full"], rel=1e-6)
