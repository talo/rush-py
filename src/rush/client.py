import inspect
import json
import platform
import random
import re
import sys
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import version as pkg_version
from io import BytesIO
from os import getenv
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Any, Literal, NewType, TypeAlias, TypeGuard

import requests
import zstandard as zstd
from gql import Client, FileVar, gql
from gql.transport.requests import RequestsHTTPTransport

from ._utils import clean_dict, optional_str

INITIAL_POLL_INTERVAL = 0.5

MAX_POLL_INTERVAL = 30

BACKOFF_FACTOR = 1.5

RunID = NewType("RunID", str)

#: UUID identifying an object in the Rush object store.
ObjectID = NewType("ObjectID", str)

_dotenv_cache: dict[str, str] | None = None


def _load_dotenv() -> dict[str, str]:
    global _dotenv_cache
    if _dotenv_cache is not None:
        return _dotenv_cache

    _dotenv_cache = {}

    # Walk up from cwd to find the nearest .env, then fall back to ~/.rush/.env
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / ".env")
    candidates.append(Path.home() / ".rush" / ".env")

    for path in candidates:
        if path.is_file():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if (
                        len(value) >= 2
                        and value[0] in ('"', "'")
                        and value[-1] == value[0]
                    ):
                        value = value[1:-1]
                    _dotenv_cache.setdefault(key, value)
            break
    return _dotenv_cache


def _get_env(key: str) -> str | None:
    value = getenv(key)
    if value is not None:
        return value
    return _load_dotenv().get(key)


GRAPHQL_ENDPOINT = getenv(
    "RUSH_ENDPOINT",
    "https://tengu-server-prod-api-519406798674.asia-southeast1.run.app",
)

DEFAULT_TARGETS = (
    ("Bullet", "Bullet2", "Bullet3")
    if "staging" in GRAPHQL_ENDPOINT
    else ("Bullet", "Bullet3")
)


def _get_api_key() -> str:
    api_key = _get_env("RUSH_TOKEN")
    if not api_key:
        raise Exception("RUSH_TOKEN must be set")
    return api_key


def _get_project_id() -> str:
    project_id = _get_env("RUSH_PROJECT")
    if not project_id:
        raise Exception("RUSH_PROJECT must be set")
    return project_id


MODULE_OVERRIDES = getenv("RUSH_MODULE_LOCK")
MODULE_OVERRIDES = json.loads(MODULE_OVERRIDES) if MODULE_OVERRIDES else {}

MODULE_LOCK = (
    {
        # staging
        "auto3d_rex": "github:talo/tengu-auto3d/88c2fdc505f206463a9c60519273563b1dddabc9#auto3d_rex",
        "boltz2_rex": "github:talo/tengu-boltz2/76df0b4b4fa42e88928a430a54a28620feef8ea8#boltz2_rex",
        "exess_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_qmmm_rex",
        "mmseqs2_rex": "github:talo/tengu-colabfold/749a096d082efdac3ac13de4aaa98aee3347d79d#mmseqs2_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/64dc3a9f37384508498c087f4c919673616302cc#prepare_protein_rex",
    }
    if "staging" in GRAPHQL_ENDPOINT
    else {
        # prod
        "auto3d_rex": "github:talo/tengu-auto3d/88c2fdc505f206463a9c60519273563b1dddabc9#auto3d_rex",
        "boltz2_rex": "github:talo/tengu-boltz2/76df0b4b4fa42e88928a430a54a28620feef8ea8#boltz2_rex",
        "exess_rex": "github:talo/tengu-exess/b667752cc767a223126184a3e78485a465a32aea#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/b667752cc767a223126184a3e78485a465a32aea#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/b667752cc767a223126184a3e78485a465a32aea#exess_qmmm_rex",
        "mmseqs2_rex": "github:talo/tengu-colabfold/0b6ca8b9dc97fc6380d334169a6faae51d85fac7#mmseqs2_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/64dc3a9f37384508498c087f4c919673616302cc#prepare_protein_rex",
    }
) | MODULE_OVERRIDES

# SDK session ID — unique per process
_SDK_SESSION_ID = str(uuid.uuid4())


def _infer_sdk_function() -> str | None:
    """Infer which SDK function called _submit_rex() by walking the stack."""
    try:
        for frame_info in inspect.stack():
            module_path = frame_info.filename
            # Look for files in the rush package (but not client.py itself)
            if "/rush/" in module_path and "client.py" not in module_path:
                module_name = Path(module_path).stem
                func_name = frame_info.function
                return f"{module_name}.{func_name}"
    except Exception:
        pass
    return None


def _get_sdk_tags(rex: str) -> list[str]:
    """Generate SDK metadata tags for run submission."""
    tags = []

    # Source tag (always rushpy for SDK submissions)
    tags.append("source=rushpy")

    # SDK version
    try:
        version = pkg_version("rush-py")
        tags.append(f"sdk_version={version}")
    except Exception:
        pass

    # SDK session ID (unique per process)
    tags.append(f"sdk_session_id={_SDK_SESSION_ID}")

    # Python version
    tags.append(f"sdk_python={platform.python_version()}")

    # Platform (OS/arch)
    machine = platform.machine()
    system = platform.system().lower()
    tags.append(f"sdk_platform={system}/{machine}")

    # Infer which SDK function submitted this run
    sdk_function = _infer_sdk_function()
    if sdk_function:
        tags.append(f"sdk_function={sdk_function}")

    return tags


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
                headers={"Authorization": f"Bearer {_get_api_key()}"},
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
            target=optional_str(
                self.target or random.choice(DEFAULT_TARGETS),
                "ModuleInstanceTarget::",
            ),
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


def upload_object(input: Path | str | dict[str, Any]):
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
    if isinstance(input, dict):
        t_f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(input, t_f)
        t_f.close()
        return upload_object(t_f.name)

    if isinstance(input, str):
        filepath = Path(input)
    else:
        filepath = input
    with filepath.open(mode="rb") as f:
        project_id = _get_project_id()
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


def _extract_object_archive(data: bytes) -> bytes:
    decompressed = zstd.ZstdDecompressor().decompress(data, max_output_size=int(1e9))
    with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
        tar_filenames = tar.getnames()

        # Handle empty tar archives
        if not tar_filenames:
            raise ValueError("Tar archive is empty - no files to extract")

        # Extract the appropriate file:
        # - If 1 file: extract that file
        # - If 2+ files: extract index 1 (skip index 0, which is often metadata)
        file_index = 1 if len(tar_filenames) >= 2 else 0
        member = tar.getmember(tar_filenames[file_index])

        # If we selected a directory, find the first actual file instead
        if member.isdir():
            file_index = None
            for i, name in enumerate(tar_filenames):
                m = tar.getmember(name)
                if not m.isdir():
                    file_index = i
                    break
            if file_index is None:
                raise ValueError(
                    "Tar archive contains only directories, no files to extract"
                )

        extracted_file = tar.extractfile(tar_filenames[file_index])
        if extracted_file is None:
            raise ValueError(
                f"Failed to extract file '{tar_filenames[file_index]}' from tar archive"
            )

        return extracted_file.read()


def fetch_object(path: str, extract: bool = False):
    """
    Fetch the contents of the given Rush object store path directly into memory.

    Be careful: if the contents are too large, they might not fit into memory.

    Args:
        path: The Rush object store path to fetch.
        extract: Automatically extract tar.zst archives in memory before returning.
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
        data = response.content
        return _extract_object_archive(data) if extract else data

    raise Exception(f"Object at path {path} has neither contents nor URL")


def _json_content_name(prefix: str, d: dict) -> str:
    payload = json.dumps(clean_dict(d), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_OID, payload)}"


def save_json(
    d: dict[str, Any], filepath: Path | str | None = None, name: str | None = None
):
    """
    Save a JSON file into the workspace folder.
    Convenient for saving non-object JSON output from a module run alongside
    the object outputs.
    """
    if filepath is not None and name is None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
    elif filepath is None and name is not None:
        project_id = _get_project_id()
        filepath = _get_opts().workspace_dir / project_id / f"{name}.json"
    else:
        raise Exception("Must specify either filepath or name")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(clean_dict(d), f, indent=2)
    return filepath


@dataclass(frozen=True)
class RushObject:
    """Reference to an object in the Rush object store."""

    #: UUID path in the object store.
    path: ObjectID
    #: Size in bytes.
    size: int
    #: Storage format.
    format: Literal["Json", "Bin"]

    @classmethod
    def from_dict(cls, d: dict) -> "RushObject":
        """Construct from a raw GraphQL output dict.

        Requires ``path``, ``size``, and ``format`` keys.
        """
        try:
            return cls(
                path=ObjectID(d["path"]),
                size=d["size"],
                format=d["format"],
            )
        except KeyError as e:
            raise ValueError(
                f"RushObject dict missing required key {e}; got keys: {list(d.keys())}"
            ) from e

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "size": self.size, "format": self.format}

    def save(
        self,
        filepath: Path | str | None = None,
        name: str | None = None,
        ext: str | None = None,
        extract: bool = False,
    ) -> Path:
        """Download this object and save to the workspace.

        The file type is derived from :attr:`format` automatically.
        Pass *ext* to override the file extension (e.g. ``"hdf5"``,
        ``"a3m"``).
        """
        if ext is None:
            ext = self.format.lower()

        if filepath is not None and name is None:
            if isinstance(filepath, str):
                filepath = Path(filepath)
        elif filepath is None and name is not None:
            project_id = _get_project_id()
            filepath = _get_opts().workspace_dir / project_id / (f"{name}." + ext)
        elif filepath is None and name is None:
            project_id = _get_project_id()
            filepath = _get_opts().workspace_dir / project_id / (f"{self.path}." + ext)
        else:
            raise Exception("Cannot specify both filepath and name")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.format == "Json":
            d = json.loads(fetch_object(self.path).decode())
            with open(filepath, "w") as f:
                json.dump(clean_dict(d), f, indent=2)
        else:
            data = fetch_object(self.path, extract=extract)
            with open(filepath, "wb") as f:
                f.write(data)

        return filepath


def save_object(
    path: str,
    filepath: Path | str | None = None,
    name: str | None = None,
    type: Literal["json", "bin"] | None = None,
    ext: str | None = None,
    extract: bool = False,
) -> Path:
    """Save a Rush object store path to the workspace.

    Prefer :meth:`RushObject.save` when you have a ``RushObject``.
    This function infers the format from the *type* parameter.
    """
    if type is None:
        type = "json"
    format: Literal["Json", "Bin"] = "Json" if type == "json" else "Bin"
    obj = RushObject(path=ObjectID(path), size=0, format=format)
    return obj.save(filepath=filepath, name=name, ext=ext, extract=extract)


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


def _format_failed_run(message: str, trace: str = "") -> str:
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

    # This shouldn't be necessary, but we'll leave it in case we have
    # a module that still manually places stdout and stderr in the trace.
    stdout_match = re.search(r'stdout: Some\("(.*?)"\)', trace, re.DOTALL)
    stderr_match = re.search(r'stderr: Some\("(.*?)"\)', trace, re.DOTALL)
    trace_without_streams = re.sub(
        r'stdout: Some\(".*?"\)|stderr: Some\(".*?"\)',
        "",
        trace,
        flags=re.DOTALL,
    )
    trace_lines = [line.rstrip() for line in trace_without_streams.splitlines()]
    trace_lines = [line for line in trace_lines if line.strip()]
    lines = [message]
    if trace_lines:
        lines.append("Trace:")
        for line in trace_lines:
            lines.append(f"  {line}")

    if stdout_match:
        lines.append("stdout:")
        for line in stdout_match.group(1).split("\n"):
            lines.append(f"  {line}")
    if stderr_match:
        lines.append("stderr:")
        for line in stderr_match.group(1).split("\n"):
            lines.append(f"  {line}")

    if trace_lines or stdout_match or stderr_match:
        lines.append("")

    return "\n".join(lines)


type RunStatus = Literal["pending", "running", "done", "error", "cancelled", "draft"]


@dataclass
class RushRunError(Exception):
    """Raised when a Rush run fails during collection."""

    message: str
    trace: str = ""

    def __str__(self) -> str:
        return _format_failed_run(self.message, self.trace)


def _build_filters(
    *,
    name: str | None,
    name_contains: str | None,
    status: RunStatus | list[RunStatus] | None,
    tags: list[str] | None,
) -> dict | None:
    """Build the GraphQL filter input from Python arguments."""
    filters: dict[str, Any] = {
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
) -> list[RunID]:
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

    run_ids: list[RunID] = []
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
        run_ids.extend(RunID(node["id"]) for node in runs_data["nodes"])

        if limit and len(run_ids) >= limit:
            return run_ids[:limit]

        if not runs_data["page_info"]["has_next_page"]:
            break

        cursor = runs_data["page_info"]["end_cursor"]

    return run_ids


def delete_run(run_id: str | RunID) -> None:
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


def _submit_rex(project_id: str, rex: str, run_opts: RunOpts = RunOpts()) -> RunID:
    # Auto-generate SDK metadata tags
    auto_tags = _get_sdk_tags(rex)

    # Merge auto-tags with user-provided tags (user tags take priority)
    if run_opts.tags:
        merged_tags = run_opts.tags + auto_tags
    else:
        merged_tags = auto_tags

    # Create a new RunOpts with merged tags
    run_opts_with_tags = RunOpts(
        name=run_opts.name,
        description=run_opts.description,
        tags=merged_tags,
        email=run_opts.email,
    )

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
        k: v for k, v in asdict(run_opts_with_tags).items() if v is not None
    }

    result = _get_client().execute(mutation)
    run_id = RunID(result["eval"]["id"])
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
class RushRunInfo:
    """
    Print it out to see a nicely-formatted summary of a run!
    """

    id: RunID
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
            f"RushRunInfo: {self.name or '(unnamed)'}",
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


def fetch_run_info(run_id: str | RunID) -> RushRunInfo | None:
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

    return RushRunInfo(**result["run"] | {"id": RunID(str(run_id))})


def _poll_run(run_id: str | RunID, max_wait_time) -> tuple[str, bool]:
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
    module_instance_created = False
    while time.time() - start_time < max_wait_time:
        time.sleep(poll_interval)

        result = _get_client().execute(query)
        status = result["run"]["status"]
        module_instances = result["run"]["module_instances"]["nodes"]
        if module_instances:
            module_instance_created = True
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
            return status, module_instance_created

        poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)

    return status, module_instance_created


def collect_run(run_id: str | RunID, max_wait_time: int = 3600):
    """
    Wait until the run finishes and return its outputs.

    Raises:
        RushRunError: If the run times out, is cancelled, or finishes with an error.
    """
    status, module_instance_created = _poll_run(run_id, max_wait_time)
    if status not in ["cancelled", "error", "done"]:
        err = f"Run timed out: did not complete within {max_wait_time} seconds"
        raise RushRunError(err)

    run = _fetch_results(run_id)
    if run["status"] == "cancelled":
        run_error = RushRunError(f"Cancelled: {run['result']}", run["trace"] or "")
        print(run_error, file=sys.stderr)
        raise run_error
    elif run["status"] == "error":
        run_error = RushRunError(f"Error: {run['result']}", run["trace"] or "")
        print(run_error, file=sys.stderr)
        raise run_error
    elif run["status"] == "done" and not module_instance_created:
        print("Restored already-completed run", file=sys.stderr)

    result = run["result"]

    def is_result_type(result: Any) -> TypeGuard[dict[str, Any]]:
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
            run_error = RushRunError(f"Error: {result['Err']}", run["trace"] or "")
            print(run_error, file=sys.stderr)
            raise run_error

    # inner error: for logic-level failures (may not exist, but should)
    if is_result_type(result):
        if "Ok" in result:
            result = result["Ok"]
        elif "Err" in result:
            run_error = RushRunError(f"Error: {result['Err']}", run["trace"] or "")
            print(run_error, file=sys.stderr)
            raise run_error

    return result


#: All self-explanatory: pending runs are queued for submission to a target.
RunStatus: TypeAlias = Literal[
    "pending", "running", "done", "error", "cancelled", "draft"
]


#: Valid values for the `target` field of `RunSpec`.
Target: TypeAlias = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

#: Valid values for the `storage_units` field of `RunSpec`.
StorageUnit: TypeAlias = Literal["KB", "MB", "GB"]
