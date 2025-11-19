import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from os import getenv
from pathlib import Path
from string import Template
from typing import Literal

import requests
from gql import Client as GqlClient, FileVar, gql
from gql.transport.requests import RequestsHTTPTransport

from .utils import clean_dict, optional_str

RUSH_URL = getenv(
    "RUSH_URL",
    "https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app",
)

MODULE_OVERRIDES = getenv("RUSH_MODULE_LOCK")
MODULE_OVERRIDES = json.loads(MODULE_OVERRIDES) if MODULE_OVERRIDES else {}

MODULE_LOCK = (
    {
        # staging
        "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/af035b062ed491c09dba9c558a8418f3482fc924#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/af035b062ed491c09dba9c558a8418f3482fc924#exess_qmmm_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/085222a5eec82dcb1dacf2b3c497e8907bd6790e#prepare_protein_rex",
    }
    if "staging" in RUSH_URL
    else {
        # prod
        "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/61b1874f8df65a083e9170082250473fd8e46978#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/61b1874f8df65a083e9170082250473fd8e46978#exess_qmmm_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/33575f99ec89bd6e28b42ac28d8e992ca137d9a7#prepare_protein_rex",
    }
) | MODULE_OVERRIDES

type TargetT = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

type StorageUnitT = Literal["KB", "MB", "GB"]


@dataclass
class RunSpec:
    target: TargetT | None = None
    walltime: str | None = None
    storage: int | None = 10
    storage_units: StorageUnitT | None = "MB"
    cpus: int | None = None
    gpus: int | None = None
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


@dataclass
class RunOpts:
    """
    The name of the run will show up as the name (i.e. title) of the run in the Rush UI.
    The description currently doesn't show up anywhere.
    The tags will also show up in the Rush UI and will (eventually) allow for run searching and filtering.
    The email flag, if set to True, will cause an email to be sent to you upon run completion.
    """

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    email: bool | None = None


class Client:
    gql_client: GqlClient
    initial_poll_interval: float
    max_poll_interval: float
    backoff_factor: float
    max_wait_time: int

    def __init__(self,
            api_url: str,
            api_token: str,
            initial_poll_interval: float = 0.5, 
            max_poll_interval: float = 30, 
            backoff_factor: float = 1.5, 
            max_wait_time: int = 3600):

        self.gql_client = GqlClient(
            transport=RequestsHTTPTransport(
                url=RUSH_URL,
                headers={"Authorization": f"Bearer {api_token}"},
            )
        )
        self.initial_poll_interval = initial_poll_interval
        self.max_poll_interval = max_poll_interval
        self.backoff_factor = backoff_factor
        self.max_wait_time = max_wait_time

    def upload_object(self, project_id: str, filepath: Path | str):
        mutation = gql("""
            mutation UploadObject($file: Upload!, $typeinfo: Json!, $format: ObjectFormatEnum!, $project_id: String) {
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
        """)
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open(mode="rb") as f:
            if filepath.suffix == ".json":
                mutation.variable_values = {
                    "file": FileVar(f),
                    "format": "json",
                    "typeinfo": {
                        "k": "record",
                        "t": {},
                    },
                    "project_id": project_id,
                }
            else:
                mutation.variable_values = {
                    "file": FileVar(f),
                    "format": "bin",
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
            result = self.gql_client.execute(mutation, upload_files=True)

        obj = result["upload_object"]["object"]
        return obj


    def download_object(self, path: str):
        # TODO: enforce UUID type
        query = gql("""
            query GetObject($path: String!) {
                object_path(path: $path) {
                    url
                    object {
                        format
                        size
                    }
                }
            }
        """)
        query.variable_values = {"path": path}

        result = self.gql_client.execute(query)
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


    def save_object(self, path):
        qm_output_json = json.loads(self.download_object(path).decode())
        out_path = f"{path}.json"
        with open(out_path, "w") as f:
            json.dump(clean_dict(qm_output_json), f, indent=2)
        return out_path


    def fetch_results(self, run_id: str):
        query = gql("""
            query GetResults($id: String!) {
                run(id: $id) {
                    status
                    result
                    trace
                }
            }
        """)
        query.variable_values = {"id": run_id}

        result = self.gql_client.execute(query)
        return result["run"]


    def submit_rex(self, project_id: str, rex: str, run_opts: RunOpts = RunOpts()):
        mutation = gql("""
            mutation EvalRex($input: CreateRun!) {
                eval(input: $input) {
                    id
                    status
                    created_at
                }
            }
        """)
        mutation.variable_values = {
            "input": {
                "rex": rex,
                "module_lock": MODULE_LOCK,
                "draft": False,
                "project_id": project_id,
            },
        }
        mutation.variable_values["input"] |= {
            k: v for k, v in asdict(run_opts).items() if v is not None
        }

        result = self.gql_client.execute(mutation)
        run_id = result["eval"]["id"]
        created_at = result["eval"]["created_at"].split(".")[0]
        print(f"Run submitted @ {created_at} with ID: {run_id}", file=sys.stderr)
        return run_id


    def collect_run(self, run_id: str):
        query = gql("""
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
        """)
        query.variable_values = {"id": run_id}

        start_time = time.time()
        poll_interval = self.initial_poll_interval
        last_status = None
        while time.time() - start_time < self.max_wait_time:
            time.sleep(poll_interval)

            result = self.gql_client.execute(query)
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
                    print(f"• {curr_status:11} @ {curr_status_time}", file=sys.stderr)
                    poll_interval = self.initial_poll_interval
                    last_status = curr_status
                poll_interval = min(poll_interval * self.backoff_factor, self.max_poll_interval)
            else:
                poll_interval = min(poll_interval * self.backoff_factor, 2)

            if status in ["done", "error", "cancelled"]:
                if not last_status:
                    print("Restored already-completed run", file=sys.stderr)
                return self.fetch_results(run_id)

            poll_interval = min(poll_interval * self.backoff_factor, self.max_poll_interval)

        raise Exception(f"Run timed out: did not complete within {self.max_wait_time} seconds")



def print_run_trace(result):
    print(f"Error: {result['result']}", file=sys.stderr)

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
        print("stdout:", file=sys.stderr)
        for line in stdout_content.split("\n"):
            print(f"  {line}", file=sys.stderr)
    stderr_match = re.search(r'stderr: Some\("(.*?)"\)', trace, re.DOTALL)
    if stderr_match:
        stderr_content = stderr_match.group(1)
        print("stderr:", file=sys.stderr)
        for line in stderr_content.split("\n"):
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)





