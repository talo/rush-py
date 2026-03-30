import itertools
import sys

from rush import RunOpts, auto3d
from tests._module_test_utils import assert_run_collects_and_caches


def test_auto3d():
    run = auto3d.generate(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"],
        k=5,
        run_opts=RunOpts(
            name="Rush-Py Test Auto3D 01",
            tags=["rush-py", "test"],
        ),
    )
    assert_run_collects_and_caches(run, auto3d.ResultRef)

    fetched = [
        list(item) if not isinstance(item, str) else item for item in run.fetch()
    ]
    assert len(fetched) == 2
    assert isinstance(fetched[1], str)

    successful_results = fetched[0]
    assert isinstance(successful_results, list)
    assert len(successful_results) == 5
    assert all(isinstance(result, auto3d.Result) for result in successful_results)
    assert all(result.stats.converged for result in successful_results)
    for i, result in enumerate(successful_results):
        print(f"Conformer {i}:", file=sys.stderr)
        for atom, coords in zip(
            result.conformer.topology.symbols,
            itertools.batched(result.conformer.topology.geometry, 3),
        ):
            print(f"  {atom} {coords}", file=sys.stderr)
        print(f"  {result.stats}", file=sys.stderr)
    print("", file=sys.stderr)

    saved = [list(item) if not isinstance(item, str) else item for item in run.save()]
    assert len(saved) == 2
    assert isinstance(saved[1], str)
    assert all(isinstance(result, auto3d.ResultPaths) for result in saved[0])
    for result in saved[0]:
        assert result.conformer.topology.exists()
        assert result.conformer.residues.exists()
        assert result.conformer.chains.exists()
        assert result.stats.exists()
