import re
import time
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from string import Template
from typing import Literal

import requests
from gql import Client, FileVar, gql
from gql.transport.requests import RequestsHTTPTransport

from .utils import optional_str

INITIAL_POLL_INTERVAL = 0.5
MAX_POLL_INTERVAL = 30
BACKOFF_FACTOR = 1.5
MAX_WAIT_TIME = 3600


GRAPHQL_ENDPOINT = getenv(
    "RUSH_ENDPOINT",
    "https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app",
)

API_KEY = getenv("RUSH_TOKEN")
PROJECT_ID = getenv("RUSH_PROJECT")
MODULE_LOCK = {
    "exess_rex": "github:talo/tengu-exess/561ffc58bb71b4454b9ada839867cf9f6730d685#exess_rex",
    "exess_qmmm_rex": "github:talo/tengu-exess/561ffc58bb71b4454b9ada839867cf9f6730d685#exess_qmmm_rex",
    "exess_geo_opt_rex": "github:talo/tengu-exess/25586c8f53986cfc49967cfbcef207091c95ff3f#exess_geo_opt_rex",
    "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
}

if not API_KEY:
    raise Exception("RUSH_TOKEN must be set")

if not PROJECT_ID:
    raise Exception("RUSH_PROJECT must be set")

client = Client(
    transport=RequestsHTTPTransport(
        url=GRAPHQL_ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ),
)

type TargetT = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

type StorageUnitT = Literal["KB", "MB", "GB"]


@dataclass
class RunSpec:
    target: TargetT | None = None
    walltime: str | None = None
    storage: int | None = 10
    storage_units: StorageUnitT | None = "MB"
    cpus: int | None = None
    gpus: int | None = 1
    nodes: int | None = None

    def to_rex(self):
        return Template(
            """RunSpec {
        resources = Resources {
          walltime = $walltime,
          storage = $storage,
          storage_units = $storage_units,
          storage_mounts = None,
          cpus = $cpus,
          mem = None,
          mem_units = None,
          gpus = $gpus,
          gpu_mem = None,
          gpu_mem_units = None,
          nodes = $nodes,
          internet_access = None,
        },
        target = $target
      }"""
        ).substitute(
            walltime=optional_str(self.walltime),
            storage=optional_str(self.storage),
            storage_units=optional_str(self.storage_units, "MemUnits::"),
            cpus=optional_str(self.cpus),
            gpus=optional_str(self.gpus),
            nodes=optional_str(self.nodes),
            target=optional_str(self.target, "ModuleInstanceTarget::"),
        )


def upload_object(project_id, filepath: Path | str):
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
    if isinstance(filepath, str):
        filepath = Path(filepath)
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


def fetch_results(run_id):
    query = gql(
        """
        query GetResults($id: String!) {
            run(id: $id) {
                status
                result
                trace
            }
        }
    """
    )
    query.variable_values = {"id": run_id}

    result = client.execute(query)
    return result["run"]


def print_run_trace(result):
    print(f"Error: {result['result']}")

    trace = result["trace"]
    trace = re.sub(
        r"\\u\{([0-9a-fA-F]+)\}",
        lambda m: chr(int(m.group(1), 16)),
        trace,
    )
    trace = trace.replace("\\n", "\n")
    trace = trace.replace('\\"', '"')
    try:
        trace = trace.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    stdout_match = re.search(r'stdout: Some\("(.*?)"\)', trace, re.DOTALL)
    if stdout_match:
        stdout_content = stdout_match.group(1)
        print("stdout:")
        for line in stdout_content.split("\n"):
            print(f"  {line}")
    stderr_match = re.search(r'stderr: Some\("(.*?)"\)', trace, re.DOTALL)
    if stderr_match:
        stderr_content = stderr_match.group(1)
        print("stderr:")
        for line in stderr_content.split("\n"):
            print(f"  {line}")
        print()


def submit_rex_nowait(project_id: str, rex: str):
    mutation = gql(
        """
        mutation EvalRex($input: CreateRun!) {
            eval(input: $input) {
                id
                status
                created_at
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
    created_at = result["eval"]["created_at"].split(".")[0]
    print(f"Run submitted @ {created_at} with ID: {run_id}")


def submit_rex(project_id: str, rex: str):
    mutation = gql(
        """
        mutation EvalRex($input: CreateRun!) {
            eval(input: $input) {
                id
                status
                created_at
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
    created_at = result["eval"]["created_at"].split(".")[0]
    print(f"Run submitted @ {created_at} with ID: {run_id}")

    query = gql(
        """
        query GetStatus($id: String!) {
            run(id: $id) {
                status
                module_instances {
                    nodes {
                        created_at
                        admitted_at
                        dispatched_at
                        queued_at
                        run_at
                        completed_at
                        deleted_at
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
        }
    """
    )
    query.variable_values = {"id": run_id}

    start_time = time.time()
    poll_interval = INITIAL_POLL_INTERVAL
    last_status = None
    while time.time() - start_time < MAX_WAIT_TIME:
        time.sleep(poll_interval)

        result = client.execute(query)
        status = result["run"]["status"]
        module_instances = result["run"]["module_instances"]["nodes"]
        if module_instances:
            curr_status = module_instances[0]["status"]
            if curr_status == "running":
                curr_status = "run"
            if (
                curr_status
                in [
                    "admitted",
                    "dispatched",
                    "queued",
                    "run",
                    "completed",
                    "deleted",
                ]
                and curr_status != last_status
            ):
                curr_status_time = module_instances[0][f"{curr_status}_at"].split(".")[
                    0
                ]
                print(f"• {curr_status:11} @ {curr_status_time}")
                poll_interval = INITIAL_POLL_INTERVAL
                last_status = curr_status
            poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)
        else:
            poll_interval = min(poll_interval * BACKOFF_FACTOR, 2)

        if status in ["done", "error", "cancelled"]:
            if not last_status:
                print("Restored already-completed run")
            return fetch_results(run_id)

        poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)

    raise Exception(f"Run timed out: did not complete within {MAX_WAIT_TIME} seconds")
