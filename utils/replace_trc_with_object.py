# The aim of this script is to take the json out of "topology" "residues" and "chains", upload them, and then replace them with the uuid paths
import argparse
import asyncio
from pathlib import Path
import json

from rush import Provider
from rush.async_utils import asyncio_run

async def run(client: Provider, path: Path):
    """
    Opens json file at
    """
    with open(path) as f:
        data = json.load(f)

        for entry in data["dependencies"]:
            if "Structure" in data["dependencies"][entry]:
                structure = data["dependencies"][entry]["Structure"]
                structure["topology"] = (await client.upload(structure["topology"], { "k":"@", "t": "string"})).object.model_dump()
                structure["residues"] = (await client.upload(structure["residues"], { "k": "@", "t": "string"})).object.model_dump()
                structure["chains"] = (await client.upload(structure["chains"], {"k": "@", "t": "string"})).object.model_dump()

    # write the data to a new file
    with open(path.with_suffix(".replaced.json"), 'w') as f:
        json.dump(data, f)


def main():
    argparser = argparse.ArgumentParser(description="Replace TRC with object")
    # data_path
    argparser.add_argument("data_path", help="The json file containing the data")
    # tengu token
    argparser.add_argument("--token", help="The Tengu token")
    # tengu url
    argparser.add_argument("--url", help="The Tengu URL")
    args = argparser.parse_args()

    client = Provider(access_token=args.token, url=args.url)


    #asyncio.run(
    asyncio_run(run(client, Path(args.data_path)))
    #)

if __name__ == "__main__":
    main()
