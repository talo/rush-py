import json
import re
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from os import getenv
from pathlib import Path
from string import Template
from typing import Literal

import requests
import zstandard as zstd
from gql import Client, FileVar, gql
from gql.transport.requests import RequestsHTTPTransport

from .utils import clean_dict, optional_str

INITIAL_POLL_INTERVAL = 0.5
MAX_POLL_INTERVAL = 30
BACKOFF_FACTOR = 1.5
MAX_WAIT_TIME = 3600


GRAPHQL_ENDPOINT = getenv(
    "RUSH_ENDPOINT",
    "https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app",
)

API_KEY = getenv("RUSH_TOKEN")
if not API_KEY:
    raise Exception("RUSH_TOKEN must be set")

PROJECT_ID = getenv("RUSH_PROJECT")
if not PROJECT_ID:
    raise Exception("RUSH_PROJECT must be set")

MODULE_OVERRIDES = getenv("RUSH_MODULE_LOCK")
MODULE_OVERRIDES = json.loads(MODULE_OVERRIDES) if MODULE_OVERRIDES else {}

MODULE_LOCK = (
    {
        # staging
        "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/af035b062ed491c09dba9c558a8418f3482fc924#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/af035b062ed491c09dba9c558a8418f3482fc924#exess_qmmm_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/f74ef68aec8839a9ac9f70441ebc0c17b814ab2f#prepare_protein_rex",
    }
    if "staging" in GRAPHQL_ENDPOINT
    else {
        # prod
        "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/61b1874f8df65a083e9170082250473fd8e46978#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/61b1874f8df65a083e9170082250473fd8e46978#exess_qmmm_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/f74ef68aec8839a9ac9f70441ebc0c17b814ab2f#prepare_protein_rex",
    }
) | MODULE_OVERRIDES


@dataclass
class RushOpts:
    workspace_dir: Path = Path.cwd()


_rush_opts: RushOpts | None = None


def _get_opts() -> RushOpts:
    global _rush_opts

    if _rush_opts is None:
        _rush_opts = RushOpts()

    return _rush_opts


def set_opts(workspace_dir: Path | None = None):
    opts = _get_opts()
    if workspace_dir is not None:
        opts.workspace_dir = workspace_dir


_rush_client: Client | None = None


def _get_client() -> Client:
    global _rush_client

    if _rush_client is None:
        _rush_client = Client(
            transport=RequestsHTTPTransport(
                url=GRAPHQL_ENDPOINT,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
        )

    return _rush_client


type TargetT = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

type StorageUnitT = Literal["KB", "MB", "GB"]


@dataclass
class RunSpec:
    target: TargetT | None = None
    walltime: int | None = None
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


def upload_object(project_id: str, filepath: Path | str):
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
        result = _get_client().execute(mutation, upload_files=True)

    obj = result["upload_object"]["object"]
    return obj


def download_object(path: str):
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

    result = _get_client().execute(query)
    obj_descriptor = result["object_path"]

    # Json
    if "contents" in obj_descriptor:
        return obj_descriptor["contents"]
    # Bin
    elif "url" in obj_descriptor:
        response = requests.get(obj_descriptor["url"])
        response.raise_for_status()
        return response.content

    raise Exception(f"Object at path {path} has neither contents nor URL")


def save_json(d: dict, filepath: Path | str | None = None, name: str | None = None):
    if filepath is not None and name is None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
    elif filepath is None and name is not None:
        filepath = _get_opts().workspace_dir / PROJECT_ID / f"{name}.json"
    else:
        raise Exception("Must specify either filepath or name")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(clean_dict(d), f, indent=2)
    return filepath


def save_object(
    path: str,
    filepath: Path | str | None = None,
    name: str | None = None,
    type: Literal["json", "bin"] | None = None,
    ext: str | None = None,
    extract: bool = False,
):
    if type is None and (ext is None or ext == "json"):
        type = "json"
    else:
        type = "bin"
    ext = type if ext is None else ext

    if filepath is not None and name is None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
    elif filepath is None and name is not None:
        filepath = _get_opts().workspace_dir / PROJECT_ID / (f"{name}." + ext)
    elif filepath is None and name is None:
        filepath = _get_opts().workspace_dir / PROJECT_ID / (f"{path}." + ext)
    else:
        raise Exception("Cannot specify both filepath or name")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if type == "json":
        d = json.loads(download_object(path).decode())
        with open(filepath, "w") as f:
            json.dump(clean_dict(d), f, indent=2)
    else:
        data = download_object(path)
        if extract:
            decompressed = zstd.ZstdDecompressor().decompress(
                data, max_output_size=int(1e9)
            )
            with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
                tar_filenames = tar.getnames()
                if len(tar_filenames) >= 2:
                    data = tar.extractfile(tar_filenames[1]).read()
        if len(tar_filenames) >= 2:
            with open(filepath, "wb") as f:
                f.write(data)
        else:
            filepath.touch()

    return filepath


def fetch_results(run_id: str):
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

    result = _get_client().execute(query)
    return result["run"]


def print_run_trace(run):
    print(f"Error: {run['result']}", file=sys.stderr)

    trace = run["trace"]
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


def submit_rex(project_id: str, rex: str, run_opts: RunOpts = RunOpts()):
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

    result = _get_client().execute(mutation)
    run_id = result["eval"]["id"]
    created_at = result["eval"]["created_at"].split(".")[0]
    print(f"Run submitted @ {created_at} with ID: {run_id}", file=sys.stderr)

    history_filepath = _get_opts().workspace_dir / "history.json"
    history_filepath.parent.mkdir(parents=True, exist_ok=True)

    matching_modules = [
        module
        for module in MODULE_LOCK
        if f"{module}_s" in rex or f"try_{module}" in rex
    ]
    if not matching_modules:
        print(
            "Error: no matching module for submission, not adding to history",
            file=sys.stderr,
        )
        return run_id
    elif len(matching_modules) > 1:
        print(
            "Error: > 1 matching module for submission, not adding to history",
            file=sys.stderr,
        )
        return run_id

    module = matching_modules[0]
    if history_filepath.exists():
        with open(history_filepath, "r") as f:
            history = json.load(f)
    else:
        history = {"instances": []}

    history["instances"].append(
        {
            "run_id": run_id,
            "run_created_at": created_at,
            "module_path": MODULE_LOCK[module],
        }
    )

    with open(history_filepath, "w") as f:
        json.dump(history, f, indent=2)

    return run_id


def poll_run(run_id: str, max_wait_time: int = MAX_WAIT_TIME):
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
    poll_interval = INITIAL_POLL_INTERVAL
    last_status = None
    while time.time() - start_time < MAX_WAIT_TIME:
        time.sleep(poll_interval)

        result = _get_client().execute(query)
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
                poll_interval = INITIAL_POLL_INTERVAL
                last_status = curr_status
            poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)
        else:
            poll_interval = min(poll_interval * BACKOFF_FACTOR, 2)

        if status in ["done", "error", "cancelled"]:
            if not last_status:
                print("Restored already-completed run", file=sys.stderr)
            return status

        poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)

    return status


def collect_run(run_id: str, max_wait_time: int = MAX_WAIT_TIME):
    status = poll_run(run_id, max_wait_time)
    if status not in ["cancelled", "error", "done"]:
        err = (f"Run timed out: did not complete within {MAX_WAIT_TIME} seconds",)
        print(err, file=sys.stderr)
        return err

    run = fetch_results(run_id)
    if run["status"] == "cancelled":
        err = f"Cancelled: {run['result']}"
        print(err, file=sys.stderr)
        return err
    elif run["status"] == "error":
        err = f"Error: {run['result']}"
        print_run_trace(run)
        return err

    result = run["result"]

    def is_result_type(result):
        return (
            isinstance(result, dict)
            and len(result) == 1
            and ("Ok" in result or "Err" in result)
        )

    # outer error: for tengu-level failures (should exist for try-prefixed rex fns)
    if is_result_type(result):
        if "Ok" in result:
            result = result["Ok"]
        elif "Err" in result:
            print(f"Error: {result['Err']}", file=sys.stderr)
            return result["Err"]

    # inner error: for logic-level failures (may not exist, but should)
    if is_result_type(result):
        if "Ok" in result:
            result = result["Ok"]
        elif "Err" in result:
            print(f"Error: {result['Err']}", file=sys.stderr)
            return result["Err"]

    if len(result) == 1:
        return result[0]
    else:
        return result
