import itertools
import sys
from pathlib import Path

from rush.auto3d import auto3d as run_auto3d
from rush.auto3d import fetch_outputs
from rush.client import RunError, RunOpts, set_opts


def test_auto3d():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    res = run_auto3d(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"],
        k=5,
        run_opts=RunOpts(
            name="Rush-Py Test Auto3D 01",
            tags=["rush-py", "test"],
        ),
        collect=True,
    )
    # Output is a list of TRC objects in memory, or a str if auto3d failed
    res = fetch_outputs(res)
    assert not isinstance(res, (str, RunError))
    assert len(res) == 2

    # res[0] expected to succeed
    assert not isinstance(res[0], RunError)
    for i, x in enumerate(res[0]):
        print(f"Conformer {i}:")
        for atom, coords in zip(
            x.conformer.topology.symbols,
            itertools.batched(x.conformer.topology.geometry, 3),
        ):
            print(f"  {str(atom)} {coords}", file=sys.stderr)
        print(f"  {x.stats}", file=sys.stderr)
    print("", file=sys.stderr)
    n = i + 1
    assert n == 5

    # res[1] expected to fail
    assert isinstance(res[1], RunError)
    print(res[1])


if __name__ == "__main__":
    test_auto3d()
