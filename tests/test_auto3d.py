import itertools
import sys

from rush.auto3d import generate
from rush.client import RunOpts


def test_auto3d():
    run = generate(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"],
        k=5,
        run_opts=RunOpts(
            name="Rush-Py Test Auto3D 01",
            tags=["rush-py", "test"],
        ),
    )
    ref = run.collect()
    res = ref.fetch()
    assert len(res) == 2

    # res[0] expected to succeed
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
    assert isinstance(res[1], str)
    print(res[1])
