import json
import tarfile
import time
from io import BytesIO
from os import getenv
from pathlib import Path
from string import Template
from typing import Literal

import cyclopts
import h5py
import requests
import zstandard as zstd
from gql import Client, FileVar, gql
from gql.transport.exceptions import TransportQueryError
from gql.transport.requests import RequestsHTTPTransport

INITIAL_POLL_INTERVAL = 0.5
MAX_POLL_INTERVAL = 30
BACKOFF_FACTOR = 1.5
MAX_WAIT_TIME = 3600


GRAPHQL_ENDPOINT = (
    "https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app"
)
API_KEY = getenv("RUSH_TOKEN") or ""
PROJECT_ID = getenv("RUSH_PROJECT") or ""
MODULE_LOCK = {
    "exess_rex": "github:talo/tengu-exess/9ccfa0a22d6395a34e03121b68fd7c4661722650#exess_rex",
}

if API_KEY == "":
    raise Exception("RUSH_TOKEN must be set")

if PROJECT_ID == "":
    raise Exception("RUSH_PROJECT must be set")

# GRAPHQL_ENDPOINT = (
#     "https://tengu-server-prod-seaography-720805281970.asia-southeast1.run.app"
# )
# API_KEY = "6b5428a1-ca95-4b0d-8c6a-d6a197b36f13"
# PROJECT_ID = "ba9a5fc0-24dc-4a51-a755-95a6b432c39f"
# MODULE_LOCK = {
#     "exess_rex": "github:talo/tengu-exess/66b121b25545069355d813c0305508e5b63251fb#exess_rex",
# }
client = Client(
    transport=RequestsHTTPTransport(
        url=GRAPHQL_ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ),
)

runspec = Template("""RunSpec {
        resources = Resources {
          walltime = None,
          storage = Some 10,
          storage_units = Some MemUnits::MB,
          storage_mounts = None,
          cpus = None,
          mem = None,
          mem_units = None,
          gpus = Some 1,
          gpu_mem = None,
          gpu_mem_units = None,
          nodes = None,
          internet_access = None,
        },
        target = $target
      }""")
      
def upload_object(project_id, filepath):
    mutation = gql(
        """
        mutation UploadObject($file: Upload!, $typeinfo: Json!, $format: ObjectFormat!, $project_id: String) {
            upload_object(file: $file, typeinfo: $typeinfo, format: $format, project_id: $project_id) {
                id
                object {
                    path
                    size
                    format
                }
                base_url
                url
            }
        }
     """
    )
    with filepath.open(mode="rb") as f:
        if filepath.suffix == ".json":
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "Json",
                "typeinfo": {
                    "k": "record",
                    "t": {},
                },
                "project_id": project_id,
            }
        else:
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "Bin",
                "typeinfo": {
                    "k": "record",
                    "t": {
                        "size": "u32",
                        "path": {
                            "k": "@",
                            "t": "$Bytes",
                        },
                    },
                    "n": "Object",
                },
                "project_id": project_id,
            }
        result = client.execute(mutation, upload_files=True)

    obj = result["upload_object"]["object"]
    print(f"Object uploaded: {obj}")
    return obj


def download_object(path):
    query = gql(
        """
        query GetObject($path: String!) {
            object_path(path: $path) {
                url
                object {
                    format
                    size
                }
            }
        }
    """
    )
    query.variable_values = {"path": path}

    result = client.execute(query)
    obj_descriptor = result["object_path"]

    # Json
    if obj_descriptor.get("contents") is not None:
        return obj_descriptor["contents"]
    # Bin
    elif obj_descriptor.get("url"):
        response = requests.get(obj_descriptor["url"])
        response.raise_for_status()
        return response.content

    raise Exception(f"Object at path {path} has neither contents nor URL")


def get_module_instance(run_id):
    query = gql(
        """
query GetModuleInstances($run_id: TextFilterInput!) {
    module_instances(filters: {run_id: $run_id}) {
        nodes {
            created_at
            admitted_at
            dispatched_at
            queued_at
            run_at
            completed_at
            deleted_at
            progress {
                n
                n_expected
                n_max
                done
            }
            status
            failure_reason
            failure_context {
                stdout
                stderr
                syserr
            }
        }
    }
}
"""
    )
    query.variable_values = {"run_id": {"eq": run_id}}

    result = client.execute(query)
    return result["module_instances"]["nodes"]


def fetch_results(run_id):
    query = gql(
        """
query GetResults($id: String!) {
    run(id: $id) {
        id
        status
        result
        stdout
    }
}
"""
    )
    query.variable_values = {"id": run_id}

    result = client.execute(query)
    return result["run"]


def submit_rex(project_id: str, rex: str):
    mutation = gql(
        """
        mutation EvalRex($input: CreateRun!) {
            eval(input: $input) {
                id
                status
            }
        }
    """
    )
    mutation.variable_values = {
        "input": {
            "rex": rex,
            "project_id": project_id,
            "module_lock": MODULE_LOCK,
            "draft": False,
        },
    }

    result = client.execute(mutation)
    run_id = result["eval"]["id"]
    print(f"Run submitted with ID: {run_id}")

    query = gql(
        """
        query GetStatus($id: String!) {
            run(id: $id) {
                status
            }
        }
    """
    )
    query.variable_values = {"id": run_id}

    start_time = time.time()
    poll_interval = INITIAL_POLL_INTERVAL
    while time.time() - start_time < MAX_WAIT_TIME:
        time.sleep(poll_interval)

        result = client.execute(query)
        status = result["run"]["status"]
        print(f"Status: {status}")
        module_instances = get_module_instance(run_id)
        if module_instances:
            print(f"Module status: {module_instances[0]}")

        if status == "done" or status == "error" or status == "cancelled":
            return fetch_results(run_id)

        poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)

    raise Exception(f"Timeout: Run did not complete within {MAX_WAIT_TIME} seconds")
