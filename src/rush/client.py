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
from typing import Literal, TypeAlias

import requests
import zstandard as zstd
from gql import Client, FileVar, gql
from gql.transport.requests import RequestsHTTPTransport

from .utils import clean_dict, optional_str

INITIAL_POLL_INTERVAL = 0.5

MAX_POLL_INTERVAL = 30

BACKOFF_FACTOR = 1.5

GRAPHQL_ENDPOINT = getenv(
    "RUSH_ENDPOINT",
    "https://tengu-server-prod-seaography-720805281970.asia-southeast1.run.app",
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
        "boltz2_rex": "github:talo/tengu-boltz2/76df0b4b4fa42e88928a430a54a28620feef8ea8#boltz2_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/f64f752732d89c47731085f1a688bfd2dee6dfc7#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/af035b062ed491c09dba9c558a8418f3482fc924#exess_qmmm_rex",
        "mmseqs2_rex": "github:talo/tengu-colabfold/749a096d082efdac3ac13de4aaa98aee3347d79d#mmseqs2_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/f74ef68aec8839a9ac9f70441ebc0c17b814ab2f#prepare_protein_rex",
    }
    if "staging" in GRAPHQL_ENDPOINT
    else {
        # prod
        "auto3d_rex": "github:talo/tengu-auto3d/ce81cfb6f4f2628cee07400992650c15ccec790e#auto3d_rex",
        "boltz2_rex": "github:talo/tengu-boltz2/76df0b4b4fa42e88928a430a54a28620feef8ea8#boltz2_rex",
        "exess_rex": "github:talo/tengu-exess/19af943399614b829a181c8620cc36e86b2705a8#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/d3d5a3dcf47b41ce3ed04fc7517bda8e375e5383#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/61b1874f8df65a083e9170082250473fd8e46978#exess_qmmm_rex",
        "mmseqs2_rex": "github:talo/tengu-colabfold/0b6ca8b9dc97fc6380d334169a6faae51d85fac7#mmseqs2_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/f74ef68aec8839a9ac9f70441ebc0c17b814ab2f#prepare_protein_rex",
    }
) | MODULE_OVERRIDES


@dataclass
class _RushOpts:
    """
    Options to configure rush-py. Can be set through the `set_opts` function.
    """

    #: The directory where the workspace resides. (Default: current working directory)
    #: The history JSON file will be written here and the
    #: run outputs will be downloaded here (nested under a project folder).
    workspace_dir: Path = Path.cwd()


_rush_opts: _RushOpts | None = None


def _get_opts() -> _RushOpts:
    global _rush_opts

    if _rush_opts is None:
        _rush_opts = _RushOpts()

    return _rush_opts


def set_opts(workspace_dir: Path | None = None):
    """
    Sets Rush options. Currently, only allows setting the workspace directory.
    """
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


type Target = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

type StorageUnit = Literal["KB", "MB", "GB"]


@dataclass
class RunSpec:
    """
    The run specification: configuration for the target and resources of a run.
    """

    #: The Rush-specified hardware that the run will be submitted to.
    #: By default, randomly chooses a cloud compute "Bullet" node of the three available.
    target: Target | None = None
    #: Max walltime in minutes for the run.
    walltime: int | None = None
    #: Max storage in the specified storage units for the run.
    storage: int | None = 10
    #: The storage units for the run.
    storage_units: StorageUnit | None = "MB"
    #: The number of CPUs for the run. Default is module-specific.
    cpus: int | None = None
    #: The number of GPUs for the run. Default is module-specific.
    gpus: int | None = None
    #: The number of nodes for the run. Only relevant for supercomputer targets.
    #: Default is module-specific.
    nodes: int | None = None

    def _to_rex(self):
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
    The description currently doesn't show up anywhere.
    The tags will also show up in the Rush UI and will (eventually) allow for run searching and filtering.
    The email flag, if set to True, will cause an email to be sent to you upon run completion.
    """

    #: Shows up as the name (i.e. title) of the run in the Rush UI.
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    email: bool | None = None


def upload_object(filepath: Path | str):
    """
    Upload an object at the filepath to the current project. Usually not necessary; the
    module functions should handle this automatically.
    """
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
                "project_id": PROJECT_ID,
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
                "project_id": PROJECT_ID,
            }
        result = _get_client().execute(mutation, upload_files=True)

    obj = result["upload_object"]["object"]
    return obj


def download_object(path: str):
    """
    Downloads the contents of the given Rush object store path directly into a variable.
    Be careful, if the contents are too large it might not fit into memory!
    """
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
    """
    Save a JSON file into the workspace folder.
    Convenient for saving non-object JSON output from a module run alongside
    the object outputs.
    """
    if filepath is not None and name is None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
    elif filepath is None and name is not None and PROJECT_ID is not None:
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
    """
    Saves the contents of the given Rush object store path into the workspace folder.
    Provides a variety of naming schemes, and supports automatically extracting tar.zst
    archives (which are sometimes used for module outputs).

    Note:
        The `filepath` and `name` parameters are mutually exculsive.

    Args:
        path: The Rush object store path to save.
        filepath: Overrides the path to save to.
        name: Sets the name of the file to save to.
        type: Manually specify the type of object (usually not necessary).
        ext: Manually the filetype extension to use (otherwise, based on `type`).
        extract: Automatically extract tar.zst files before saving.
    """
    if type is None and (ext is None or ext == "json"):
        type = "json"
    else:
        type = "bin"
    ext = type if ext is None else ext

    if filepath is not None and name is None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
    elif filepath is None and name is not None and PROJECT_ID is not None:
        filepath = _get_opts().workspace_dir / PROJECT_ID / (f"{name}." + ext)
    elif filepath is None and name is None and PROJECT_ID is not None:
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


def _fetch_results(run_id: str):
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


def _print_run_trace(run):
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


type RunStatus = Literal["pending", "running", "done", "error", "cancelled", "draft"]


def _build_filters(
    *,
    name: str | None,
    name_contains: str | None,
    status: RunStatus | list[RunStatus] | None,
    tags: list[str] | None,
) -> dict | None:
    """Build the GraphQL filter input from Python arguments."""
    filters = {
        # We don't want to show deleted runs
        "deleted_at": {"is_null": True},
    }

    if name is not None:
        filters["name"] = {"ci_eq": name}
    elif name_contains is not None:
        filters["name"] = {"ilike": f"%{name_contains}%"}

    if status is not None:
        if isinstance(status, list):
            filters["status"] = {"is_in": status}
        else:
            filters["status"] = {"eq": status}

    if tags is not None:
        filters["tags"] = {"array_contains": tags}

    return filters if filters else None


def fetch_runs(
    *,
    name: str | None = None,
    name_contains: str | None = None,
    status: RunStatus | list[RunStatus] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """
    Query runs and return their IDs.

    Args:
        name: Filter by exact run name (case-insensitive).
        name_contains: Filter by runs whose name contains this substring.
        status: Filter by status. Can be a single status or a list of statuses.
        tags: Filter by tags. Returns runs that have ALL specified tags.
        limit: Maximum number of runs to return. If None, returns all matching runs.

    Returns:
        A list of run IDs matching the filters.
    """
    query = gql("""
        query GetRuns($filters: RunFilterInput, $pagination: PaginationInput) {
            runs(filters: $filters, pagination: $pagination) {
                page_info {
                    has_next_page
                    end_cursor
                }
                nodes {
                    id
                }
            }
        }
    """)

    filters = _build_filters(
        name=name,
        name_contains=name_contains,
        status=status,
        tags=tags,
    )

    run_ids = []
    cursor = None
    page_limit = min(limit, 100) if limit else 100

    while True:
        if cursor:
            pagination = {"cursor": {"cursor": cursor, "limit": page_limit}}
        else:
            pagination = {"offset": {"offset": 0, "limit": page_limit}}

        query.variable_values = {"filters": filters, "pagination": pagination}
        result = _get_client().execute(query)

        runs_data = result["runs"]
        run_ids.extend(node["id"] for node in runs_data["nodes"])

        if limit and len(run_ids) >= limit:
            return run_ids[:limit]

        if not runs_data["page_info"]["has_next_page"]:
            break

        cursor = runs_data["page_info"]["end_cursor"]

    return run_ids


def delete_run(run_id: str) -> None:
    """
    Delete a run by ID.
    """
    query = gql("""
        mutation DeleteRun($run_id: String!) {
            delete_run(run_id: $run_id) {
                id
            }
        }
    """)
    query.variable_values = {"run_id": run_id}

    _get_client().execute(query)


def _submit_rex(project_id: str, rex: str, run_opts: RunOpts = RunOpts()):
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


@dataclass
class RushRun:
    """
    Print it out to see a nicely-formatted summary of a run!
    """

    id: str
    created_at: str
    updated_at: str
    status: str
    deleted_at: str | None = None
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    result: dict | None = None
    trace: dict | None = None
    stdout: str | None = None

    def __str__(self) -> str:
        lines = [
            f"RushRun: {self.name or '(unnamed)'}",
            f"  id:          {self.id}",
            f"  status:      {self.status}",
            f"  created_at:  {self.created_at}",
            f"  updated_at:  {self.updated_at}",
        ]
        if self.deleted_at:
            lines.append(f"  deleted_at:  {self.deleted_at}")
        if self.description:
            lines.append(f"  description: {self.description}")
        if self.tags:
            lines.append(f"  tags:        {', '.join(self.tags)}")
        return "\n".join(lines)


def fetch_run_info(run_id: str) -> RushRun | None:
    """
    Fetch all info for a run by ID.

    Returns `None` if the run doesn't exist.
    """
    query = gql("""
        query GetRun($id: String!) {
            run(id: $id) {
                created_at
                deleted_at
                updated_at
                name
                description
                tags
                result
                status
                trace
                stdout
            }
        }
    """)
    query.variable_values = {"id": run_id}

    result = _get_client().execute(query)
    if result["run"] is None:
        return None

    return RushRun(**result["run"] | {"id": run_id})


def _poll_run(run_id: str, max_wait_time):
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
    while time.time() - start_time < max_wait_time:
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


def collect_run(run_id: str, max_wait_time: int = 3600) -> dict | str:
    """
    Waits until the run finishes, or `max_wait_time` elapses, and returns either the
    actual result of the run, an error string if the run failed, or a string indicating
    that the run timed out.
    """
    status = _poll_run(run_id, max_wait_time)
    if status not in ["cancelled", "error", "done"]:
        err = f"Run timed out: did not complete within {max_wait_time} seconds"
        print(err, file=sys.stderr)
        return err

    run = _fetch_results(run_id)
    if run["status"] == "cancelled":
        err = f"Cancelled: {run['result']}"
        print(err, file=sys.stderr)
        return err
    elif run["status"] == "error":
        err = f"Error: {run['result']}"
        _print_run_trace(run)
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


#: All self-explanatory: pending runs are queued for submission to a target.
RunStatus: TypeAlias = Literal[
    "pending", "running", "done", "error", "cancelled", "draft"
]

#: Valid values for the `target` field of `RunSpec`.
Target: TypeAlias = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

#: Valid values for the `storage_units` field of `RunSpec`.
StorageUnit: TypeAlias = Literal["KB", "MB", "GB"]
