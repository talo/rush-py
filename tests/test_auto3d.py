from os import getenv
from pprint import pp

from rush_py2 import from_json
from rush_py2.auto3d import Auto3DOptions, auto3d
from rush_py2.client import Client


API_TOKEN = getenv("RUSH_API_TOKEN")
if not API_TOKEN:
    raise Exception("RUSH_API_TOKEN must be set")

PROJECT_ID = getenv("RUSH_PROJECT_ID")
if not PROJECT_ID:
    raise Exception("RUSH_PROJECT_ID must be set")

if __name__ == "__main__":

    client = Client(api_url="https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app", api_token=API_TOKEN)
    result = auto3d(
        client,
        PROJECT_ID,
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"], 
        Auto3DOptions(k=5), 
        collect=True
    )
    if result is None:
        raise Exception("No result")
    pp(from_json(result[0]), width=130, compact=True)
    print(result[1])
