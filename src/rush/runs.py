"""Run submission and lifecycle types."""

import random
import re
import sys
import time
from dataclasses import dataclass
from string import Template
from typing import (
    Any,
    Generic,
    Literal,
    NewType,
    TypeGuard,
    TypeVar,
)

from gql import gql

from ._rex import optional_str
from .session import DEFAULT_TARGETS, _get_client

INITIAL_POLL_INTERVAL = 0.5
MAX_POLL_INTERVAL = 30
BACKOFF_FACTOR = 1.5

#: String identifier for a Rush run.
RunID = NewType("RunID", str)

#: All self-explanatory: pending runs are queued for submission to a target.
type RunStatus = Literal["pending", "running", "done", "error", "cancelled", "draft"]

#: Valid values for the `target` field of `RunSpec`.
type Target = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]

#: Valid values for the `storage_units` field of `RunSpec`.
type StorageUnit = Literal["KB", "MB", "GB"]

R = TypeVar("R")


@dataclass
class RunSpec:
    """
    The run specification: configuration for the target and resources of a run.
    """

    #: The Rush-specified hardware that the run will be submitted to.
    #: By default, randomly chooses a cloud compute "Bullet" node of the available ones.
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

    def _to_rex(self) -> str:
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


class Run(Generic[R]):
    """Handle to a submitted Rush job."""

    def __init__(self, id: RunID, result_type: type[R]) -> None:
        self._id = id
        self._result_type = result_type
        self._collected: R | None = None

    @property
    def id(self) -> RunID:
        return self._id

    def collect(self, max_wait_time: int = 3600) -> R:
        if self._collected is None:
            raw = collect_run(self._id, max_wait_time=max_wait_time)
            self._collected = self._result_type.from_raw_output(raw)  # type: ignore[ty:unresolved-attribute]
        return self._collected

    def fetch(self, **kwargs: Any) -> Any:
        return self.collect().fetch(**kwargs)  # type: ignore[ty:unresolved-attribute]

    def save(self, **kwargs: Any) -> Any:
        return self.collect().save(**kwargs)  # type: ignore[ty:unresolved-attribute]

    def __repr__(self) -> str:
        return f"Run(id={self._id!r})"


@dataclass
class RunInfo:
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
    result: dict[str, Any] | None = None
    stdout: str | None = None
    trace: dict[str, Any] | None = None
    walltime: int | float | None = None
    sus: dict[str, int | float] | None = None

    def _resource_totals_complete(self) -> bool:
        return self.status in {"done", "error", "cancelled"}

    def __str__(self) -> str:
        lines = [
            f"Run info for {self.name or '(unnamed)'}",
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
        totals_suffix = "" if self._resource_totals_complete() else " (incomplete)"
        if self.walltime is not None:
            lines.append(f"  walltime:    {self.walltime}{totals_suffix}")
        if self.sus is not None:
            for target, sus in self.sus.items():
                prefix = f"{target.capitalize()} SUs:"
                lines.append(f"  {prefix:<12} {sus}{totals_suffix}")
        return "\n".join(lines)


def _total_run_walltime(
    resource_utilizations: dict[str, Any] | None,
) -> int | float | None:
    if resource_utilizations is None:
        return None

    return sum(
        utilization["walltime"]
        for utilization in resource_utilizations["nodes"]
        if utilization.get("walltime") is not None
    )


def _run_sus(
    resource_utilizations: dict[str, Any] | None,
    module_instances: dict[str, Any] | None = None,
) -> dict[str, int | float] | None:
    sus_by_target: dict[str, int | float] = {}
    for module_instance in module_instances["nodes"] if module_instances else []:
        target = module_instance.get("target")
        if target in {"gadi", "setonix"}:
            sus_by_target.setdefault(target, 0)

    for utilization in resource_utilizations["nodes"] if resource_utilizations else []:
        target = utilization.get("target")
        if target not in {"gadi", "setonix"}:
            continue

        sus_by_target.setdefault(target, 0)
        sus = utilization.get("sus")
        if sus is not None:
            sus_by_target[target] += sus

    return sus_by_target or None


def fetch_run_info(run_id: str | RunID) -> RunInfo | None:
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
                module_instances {
                    nodes {
                        target
                    }
                }
                resource_utilizations {
                    nodes {
                        target
                        walltime
                        sus
                    }
                }
            }
        }
    """)
    query.variable_values = {"id": run_id}

    result = _get_client().execute(query)
    if result["run"] is None:
        return None

    run = result["run"]
    walltime = _total_run_walltime(run.get("resource_utilizations"))
    sus = _run_sus(run.get("resource_utilizations"), run.get("module_instances"))
    return RunInfo(
        id=RunID(str(run_id)),
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        status=run["status"],
        deleted_at=run["deleted_at"],
        name=run["name"],
        description=run["description"],
        tags=run["tags"],
        result=run["result"],
        trace=run["trace"],
        stdout=run["stdout"],
        walltime=walltime,
        sus=sus,
    )


def _build_filters(
    *,
    name: str | None,
    name_contains: str | None,
    status: RunStatus | list[RunStatus] | None,
    tags: list[str] | None,
) -> dict[str, Any]:
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
        filters["status"] = (
            {"is_in": status} if isinstance(status, list) else {"eq": status}
        )

    if tags is not None:
        filters["tags"] = {"array_contains": tags}

    return filters


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
        pagination = (
            {"cursor": {"cursor": cursor, "limit": page_limit}}
            if cursor
            else {"offset": {"offset": 0, "limit": page_limit}}
        )
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


def _format_failed_run(
    title: str,
    message: str,
    trace: str = "",
    guidance: str | None = None,
) -> str:
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
    lines = [f"{title}: {message}"]
    if guidance:
        lines.append(guidance)
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


@dataclass
class RunError(Exception):
    """Base class for errors raised while collecting a Rush run."""

    message: str
    trace: str = ""

    def _title(self) -> str:
        return "Run error"

    def _guidance(self) -> str | None:
        return None

    def __str__(self) -> str:
        return _format_failed_run(
            self._title(),
            self.message,
            self.trace,
            self._guidance(),
        )


@dataclass
class RunBackendError(RunError):
    """Run failed due to Rush backend, infrastructure, or orchestration issues."""

    def _title(self) -> str:
        return "Rush backend error"

    def _guidance(self) -> str | None:
        return (
            "This indicates a Rush backend or infrastructure failure. "
            "Contact QDX or submit a bug report for assistance and resolution."
        )


@dataclass
class RunModuleError(RunError):
    """Run failed inside the module/application layer."""

    def _title(self) -> str:
        return "Run module error"

    def _guidance(self) -> str | None:
        return None


def _raise_run_error(error: RunError) -> None:
    print(error, file=sys.stderr)
    raise error


def _unwrap_result(
    result: Any,
    trace: str,
    error_type: type[RunError],
) -> Any:
    def is_result_type(value: Any) -> TypeGuard[dict[str, Any]]:
        return (
            isinstance(value, dict)
            and len(value) == 1
            and ("Ok" in value or "Err" in value)
        )

    if not is_result_type(result):
        return result

    if "Ok" in result:
        return result["Ok"]

    _raise_run_error(error_type(str(result["Err"]), trace))


def _poll_run(run_id: str | RunID, max_wait_time: int) -> tuple[str, bool]:
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
            module_instance = module_instances[0]
            curr_status = module_instance["status"]
            if curr_status == "running":
                curr_status = "run"
            if (
                curr_status
                in ["admitted", "dispatched", "queued", "run", "completed", "deleted"]
                and curr_status != last_status
            ):
                curr_status_time = module_instance[f"{curr_status}_at"].split(".")[0]
                print(f"• {curr_status:11} @ {curr_status_time}", file=sys.stderr)
                poll_interval = INITIAL_POLL_INTERVAL
                last_status = curr_status
            poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)
        else:
            poll_interval = min(poll_interval * BACKOFF_FACTOR, 2)

        if status in ["done", "error", "cancelled"]:
            return status, module_instance_created

    return status, module_instance_created


def _fetch_results(run_id: str) -> dict[str, Any]:
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


def collect_run(run_id: str | RunID, max_wait_time: int = 3600):
    """
    Wait until the run finishes and return its outputs.

    Raises:
        RunBackendError: If the run times out, is cancelled, or the Rush backend
            fails to execute it successfully.
        RunModuleError: If the module fails inside the module/application layer.
    """
    status, module_instance_created = _poll_run(run_id, max_wait_time)
    if status not in ["cancelled", "error", "done"]:
        err = f"Run timed out: did not complete within {max_wait_time} seconds"
        raise RunBackendError(err)

    run = _fetch_results(run_id)
    if run["status"] == "cancelled":
        _raise_run_error(
            RunBackendError(f"Cancelled: {run['result']}", run["trace"] or "")
        )
    elif run["status"] == "error":
        _raise_run_error(RunBackendError(str(run["result"]), run["trace"] or ""))
    elif run["status"] == "done" and not module_instance_created:
        print("Restored already-completed run", file=sys.stderr)

    result = run["result"]

    # outer error: for tengu-level failures (should exist for try-prefixed rex fns)
    result = _unwrap_result(result, run["trace"] or "", RunBackendError)

    # inner error: for logic-level failures (may not exist, but should)
    result = _unwrap_result(result, run["trace"] or "", RunModuleError)

    return result
