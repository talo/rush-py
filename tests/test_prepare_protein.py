import sys

from rush_py2.client import RunOpts, RunSpec
from rush_py2.prepare_protein import prepare_protein

if __name__ == "__main__":
    o = prepare_protein(
        "cdk2_dry_trc.json",
        run_spec=RunSpec(target="Bullet2"),
        run_opts=RunOpts(
            name="Test prepare-protein 02", tags=["rush-py2", "test", "cdk2"]
        ),
        collect=True,
    )
    print(o, file=sys.stderr)
