from os import getenv
import sys

from rush_py2.client import Client, RunOpts, RunSpec
from rush_py2.prepare_protein import prepare_protein


API_TOKEN = getenv("RUSH_API_TOKEN")
if not API_TOKEN:
    raise Exception("RUSH_API_TOKEN must be set")

PROJECT_ID = getenv("RUSH_PROJECT_ID")
if not PROJECT_ID:
    raise Exception("RUSH_PROJECT_ID must be set")


if __name__ == "__main__":
    client = Client(api_url="https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app", api_token=API_TOKEN)
    o = prepare_protein(
        client,
        PROJECT_ID,
        "cdk2_dry_trc.json",
        run_spec=RunSpec(target="Bullet2"),
        run_opts=RunOpts(
            name="Test prepare-protein 02", tags=["rush-py2", "test", "cdk2"]
        ),
        collect=True,
    )
    print(o, file=sys.stderr)
