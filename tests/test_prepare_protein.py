from rush_py2.client import RunOpts
from rush_py2.prepare_protein import prepare_protein

if __name__ == "__main__":
    o = prepare_protein(
        "cdk2_dry_trc.json",
        run_opts=RunOpts(
            name="Test prepare-protein 01",
            tags=["rush-py2", "test", "cdk2"],
            email=True,
        ),
        collect=True,
    )
    print(o)
