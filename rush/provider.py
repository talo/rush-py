from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from io import IOBase
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Generic,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

import httpx
from pydantic_core import to_jsonable_python

from rush.graphql_client.benchmark_submission import (
    BenchmarkSubmissionMeAccountProjectBenchmarkSubmission,
)
from rush.graphql_client.run import RunMeAccountProjectRun

from .async_utils import LOOP, asyncio_run, start_background_loop
from .graphql_client.argument import Argument, ArgumentArgument
from .graphql_client.arguments import (
    ArgumentsMeAccountArguments,
    ArgumentsMeAccountArgumentsEdgesNode,
    ArgumentsMeAccountArgumentsPageInfo,
)
from .graphql_client.base_model import UNSET, UnsetType
from .graphql_client.benchmark import BenchmarkBenchmark
from .graphql_client.benchmark_submissions import (
    BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsEdgesNode,
    BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsPageInfo,
)
from .graphql_client.benchmarks import (
    BenchmarksBenchmarks,
    BenchmarksBenchmarksEdgesNode,
    BenchmarksBenchmarksPageInfo,
)
from .graphql_client.client import Client
from .graphql_client.create_project import CreateProjectCreateProject
from .graphql_client.enums import (
    MemUnits,
    ModuleInstanceStatus,
    ModuleInstanceTarget,
    ObjectFormat,
)
from .graphql_client.eval import EvalEval
from .graphql_client.exceptions import GraphQLClientGraphQLMultiError
from .graphql_client.fragments import (
    ModuleFull,
    ModuleInstanceFullProgress,
    PageInfoFull,
)
from .graphql_client.input_types import (
    ArgumentInput,
    BenchmarkFilter,
    CreateMetadata,
    CreateModuleInstance,
    CreateProject,
    CreateRun,
    DateTimeFilter,
    MetadataFilter,
    ModuleFilter,
    ModuleInstanceFilter,
    ModuleInstanceStatusFilter,
    ProjectFilter,
    ResourcesInput,
    StringFilter,
    TagFilter,
    UuidFilter,
)
from .graphql_client.latest_modules import LatestModulesLatestModulesPageInfo
from .graphql_client.module_instance_details import ModuleInstanceDetailsModuleInstance
from .graphql_client.module_instance_full import ModuleInstanceFullModuleInstance
from .graphql_client.module_instances import (
    ModuleInstancesMeAccountModuleInstancesEdgesNode,
)
from .graphql_client.modules import ModulesModulesPageInfo
from .graphql_client.object_contents import ObjectContentsObjectPath
from .graphql_client.projects import (
    ProjectsMeAccountProjects,
    ProjectsMeAccountProjectsEdgesNode,
    ProjectsMeAccountProjectsPageInfo,
)
from .graphql_client.retry import RetryRetry
from .graphql_client.run_benchmark import RunBenchmarkRunBenchmark
from .graphql_client.run_module_instance import RunModuleInstanceRunModuleInstance
from .graphql_client.runs import (
    RunsMeAccountProjectRunsEdgesNode,
    RunsMeAccountProjectRunsPageInfo,
)
from .typedef import (
    SCALARS,
    RushType,
    build_typechecker,
    format_module_typedesc,
    type_from_typedef,
)

if sys.version_info >= (3, 12):
    from .types import ArgId, ModuleInstanceId, Resources, Target
else:
    from .legacy_types import ArgId, ModuleInstanceId, Resources, Target


class VirtualObject:
    path: str
    size: int
    format: ObjectFormat


@dataclass
class ModuleInstanceHistory:
    path: str
    id: ModuleInstanceId
    status: ModuleInstanceStatus
    tags: Sequence[str]


@dataclass
class History:
    tags: list[str]
    instances: list[ModuleInstanceHistory]


T = TypeVar("T")
TCo = TypeVar("TCo", covariant=True)
TInv = TypeVar("TInv", covariant=False, contravariant=False)
TPage = TypeVar("TPage", bound=PageInfoFull)


class Edge(Protocol[TInv]):
    cursor: str
    node: TInv


class Page(Protocol[TInv, TPage]):
    edges: Iterable[Edge[TInv]]
    page_info: TPage


# Represents the return type of a paged query
# Should contain a list of nodes and a page_info object
# The page_info object should contain a hasPreviousPage field
T1 = TypeVar("T1", bound=Page[Any, Any])


class ProjectPaged(
    Protocol[T1, TPage],
):
    async def __call__(
        self,
        project_id: str,
        after: Union[Optional[str], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        first: Union[Optional[int], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        **kwargs: Any,
    ) -> Page[T1, TPage]:
        ...


class Paged(
    Protocol[T1, TPage],
):
    async def __call__(
        self,
        after: Union[Optional[str], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        first: Union[Optional[int], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        **kwargs: Any,
    ) -> Page[T1, TPage]:
        ...


@dataclass
class PageVars:
    after: Union[Optional[str], UnsetType] = UNSET
    before: Union[Optional[str], UnsetType] = UNSET
    first: Union[Optional[int], UnsetType] = UNSET
    last: Union[Optional[int], UnsetType] = UNSET


class EmptyPage(Generic[T1, TPage], Page[T1, TPage]):
    # skip the type checker for this class
    # since it is only used for the empty page
    page_info: Any = PageInfoFull(hasPreviousPage=False, hasNextPage=False, startCursor=None, endCursor=None)
    edges: Sequence[Edge[T1]] = []


class RushModuleRunner(Protocol[TCo]):
    async def __call__(
        self,
        *args: Any,
        target: Target,
        resources: Resources | None = None,
        tags: list[str] | None = None,
        restore: bool | None = False,
    ) -> TCo:
        ...


def get_name_from_path(path: str):
    return path.split("#")[-1].replace("_tengu", "").replace("tengu_", "")


UI_URL_MAP = {
    "https://tengu-server-staging-edh4uref5a-as.a.run.app": "https://rush-qdx-2-staging.web.app",
    "https://tengu.qdx.ai": "https://rush.cloud",
}


class BaseProvider:
    """
    A class representing a provider for the Rush quantum chemistry workflow platform.
    """

    def __init__(
        self,
        client: Client,
        logger: logging.Logger,
        restore_by_default: bool = False,
        workspace: str | Path | None = None,
        batch_tags: Sequence[str] | None = None,
        project_id: UUID | None = None,
    ):
        """
        Initialize the RushProvider a graphql client.
        """
        self.restore_by_default = restore_by_default
        self.history = None
        self.client = client
        self.client.http_client.timeout = httpx.Timeout(60)
        self.module_paths: dict[str, str] = {}
        self.logger = logger
        self.project_id = project_id

        self.__is_blocking__ = False

        if workspace:
            self.workspace: Path | None = Path(workspace)
            if not self.workspace.exists():
                raise Exception("Workspace directory does not exist")
            if (self.workspace / "rush.lock").exists():
                self._config_dir: Path | None = self.workspace
            else:
                self._config_dir = self.workspace / ".rush"
                if not self._config_dir.exists():
                    self._config_dir.mkdir()

            self.restore(workspace)
        else:
            self.workspace = None
            self._config_dir = None

        if not self.history:
            self.history = History(tags=list(batch_tags or []), instances=[])
        self.batch_tags = list(batch_tags or [])

    @staticmethod
    def _load_history(history_file: str | Path):
        """
        Load the history from a file.
        """
        with open(history_file, "r") as f:
            json_dict = json.load(f)
            return History(
                tags=json_dict["tags"],
                instances=[ModuleInstanceHistory(**instance) for instance in json_dict["instances"]],
            )

    async def nuke(self, remote: bool = False, tags: bool = False):
        """
        Delete the workspace, and optionally the data stored for it on the server.
        """
        # first untrack the runs remotely if necessary
        if remote:
            if self.history:
                for instance in self.history.instances:
                    try:
                        await self.delete_module_instance(instance.id)
                    except Exception as e:
                        if "not found" in str(e):
                            pass
                        else:
                            raise e
            if tags:
                async for page in await self.module_instances(tags=self.batch_tags):
                    for instance in page.edges:
                        try:
                            await self.delete_module_instance(instance.node.id)
                        except Exception as e:
                            if "not found" in str(e):
                                pass
                            else:
                                raise e
        if self.workspace:
            if (self.workspace / "rush.lock").exists():
                (self.workspace / "rush.lock").unlink()
            for file in (self.workspace / ".rush").glob("*"):
                file.unlink()
            for file in (self.workspace / "objects").glob("*"):
                file.unlink()
            if (self.workspace / ".rush").exists():
                (self.workspace / ".rush").rmdir()
            if (self.workspace / "objects").exists():
                (self.workspace / "objects").rmdir()

    def restore(self, workspace: str | Path):
        """
        Restore the workspace.
        """
        self.workspace = Path(workspace)

        if (self.workspace / "rush.lock").exists():
            self._config_dir = self.workspace
        else:
            self._config_dir = self.workspace / ".rush"
            if not self._config_dir.exists():
                self._config_dir.mkdir()
        # read the workspace history file
        # if it exists, load the history
        workspace_history = self._config_dir / "history.json"
        if workspace_history.exists():
            self.history = self._load_history(workspace_history)

        if (self._config_dir / "rush.lock").exists():
            self.load_module_paths(self._config_dir / "rush.lock")

    def save(self, history_file: str | Path | None = None):
        """
        Save the workspace.
        """
        if self._config_dir is None:
            raise Exception("No workspace provided")
        if history_file is None:
            history_file = self._config_dir / "history.json"
        self.save_module_paths(self.module_paths, self._config_dir / "rush.lock")
        with open(history_file, "w") as f:
            json.dump(self.history, f, default=to_jsonable_python, indent=2)

    async def status(
        self,
        instance_ids: list[ModuleInstanceId] | None = None,
        workspace: str | Path | None = None,
        history_file: str | Path | None = None,
        group_by: Literal["tag", "path", "id"] = "id",
    ) -> dict[str, tuple[ModuleInstanceStatus, str, int]]:
        """
        Get the status of all module instances in a workspace, grouped by tag, path, or id.
        """
        if not instance_ids:
            history = self.history
            if workspace:
                if isinstance(workspace, str):
                    workspace = Path(workspace)
                history = self._load_history(workspace / "history.json")
            elif history_file:
                history = self._load_history(history_file)

            if not history:
                return {}

            instance_ids = [instance.id for instance in history.instances]

        instances: list[ModuleInstanceFullModuleInstance] = []
        async for page in await self.module_instances(ids=instance_ids):
            for instance in page.edges:
                instances.append(instance.node)

        if group_by == "id":
            return {
                str(instance.id): (
                    instance.status,
                    get_name_from_path(instance.path),
                    1,
                )
                for instance in instances
            }

        if group_by == "path":
            c = Counter([(instance.status, get_name_from_path(instance.path)) for instance in instances])
            return {f"{name} ({status})": (status, name, count) for (status, name), count in c.items()}

        if group_by == "tag":
            c = Counter([(instance.status, instance.tags) for instance in instances])
            return {f"{tag} ({status})": (status, "", count) for (status, tag), count in c.items()}

        raise Exception("Invalid group_by")

    async def _project_query_with_pagination(
        self,
        fn: ProjectPaged[T1, TPage],
        project_id: UUID,
        page_vars: PageVars,
        variables: dict[str, Any],
    ) -> AsyncIterable[Page[T1, TPage]]:
        result = await fn(str(project_id), **variables)

        page_info_res = result.page_info
        yield result or EmptyPage[T1, TPage]()

        while page_info_res.has_previous_page:
            page_vars.before = page_info_res.end_cursor
            result = await fn(
                str(project_id),
                **page_vars.__dict__,
                **variables,
            )
            page_info_res = result.page_info
            yield result or EmptyPage()
            if len(result.edges) == 0:  # type: ignore
                break

    async def _query_with_pagination(
        self,
        fn: Paged[T1, TPage],
        page_vars: PageVars,
        variables: dict[str, Any],
    ) -> AsyncIterable[Page[T1, TPage]]:
        result = await fn(**variables)

        page_info_res = result.page_info
        yield result or EmptyPage[T1, TPage]()

        while page_info_res.has_previous_page:
            page_vars.before = page_info_res.end_cursor
            result = await fn(
                **page_vars.__dict__,
                **variables,
            )
            page_info_res = result.page_info
            yield result or EmptyPage()
            if len(result.edges) == 0:  # type: ignore
                break

    async def argument(self, id: ArgId) -> ArgumentArgument:
        """
        Retrieve an argument from the database.

        :param id: The ID of the argument.
        :return: The argument.
        """
        return await self.client.argument(id)

    async def arguments(
        self,
        after: str | None = None,
        before: str | None = None,
        first: int | None = None,
        last: int | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterable[Page[ArgumentsMeAccountArguments, ArgumentsMeAccountArgumentsPageInfo]]:
        """
        Retrieve a list of arguments.
        """

        async def return_paged(
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[ArgumentsMeAccountArgumentsEdgesNode, ArgumentsMeAccountArgumentsPageInfo]:
            res = await self.client.arguments(first=first, after=after, last=last, before=before, **kwargs)
            # The types for this pass in mypy, but not in pyright
            return res.account.arguments  # type: ignore

        return self._query_with_pagination(
            return_paged,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {
                "tags": tags,
            },
        )

    async def benchmarks(
        self,
        after: str | None = None,
        before: str | None = None,
        first: int | None = None,
        last: int | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterable[Page[BenchmarksBenchmarks, BenchmarksBenchmarksPageInfo]]:
        return self._query_with_pagination(
            self.client.benchmarks,
            PageVars(after=after, before=before, first=first, last=last),
            {},
        )

    async def benchmark(
        self, id: str | None = None, name: str | None = None
    ) -> BenchmarkBenchmark | BenchmarksBenchmarksEdgesNode:
        if id:
            return await self.client.benchmark(id)
        if name:
            return (
                (
                    await self.client.benchmarks(
                        filter=BenchmarkFilter(metadata=MetadataFilter(name=StringFilter(eq=name)))
                    )
                )
                .edges[0]
                .node
            )
        else:
            raise Exception("Please provide either an id or a name")

    async def benchmark_submissions(
        self,
    ) -> AsyncIterable[
        Page[
            BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsEdgesNode,
            BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsPageInfo,
        ]
    ]:
        """
        Retrieve a list of runs.
        """
        if not self.project_id:
            raise Exception("No project ID provided")

        async def return_paged(
            project_id: str,
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[
            BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsEdgesNode,
            BenchmarkSubmissionsMeAccountProjectBenchmarkSubmissionsPageInfo,
        ]:
            res = await self.client.benchmark_submissions(
                project_id=project_id,
                first=first,
                after=after,
                last=last,
                before=before,
                **kwargs,
            )
            # The types for this pass in mypy, but not in pyright
            return res.account.project.benchmark_submissions  # type: ignore

        return self._project_query_with_pagination(return_paged, UUID(self.project_id), PageVars(), {})

    async def benchmark_submission(
        self,
        benchmark_submission_id: str,
    ) -> BenchmarkSubmissionMeAccountProjectBenchmarkSubmission:
        return (
            await self.client.benchmark_submission(project_id=self.project_id, id=benchmark_submission_id)
        ).account.project.benchmark_submission

    async def poll_benchmark_submission(
        self,
        benchmark_submission_id: str,
        with_scores: bool = False,
    ) -> BenchmarkSubmissionMeAccountProjectBenchmarkSubmission:
        submission = await self.benchmark_submission(benchmark_submission_id)
        while submission.source_run and submission.source_run.status not in [
            "DONE",
            "ERROR",
        ]:
            await asyncio.sleep(5)
            submission = await self.benchmark_submission(benchmark_submission_id)
        if with_scores:
            while not submission.scores.nodes:
                await asyncio.sleep(5)
                submission = await self.benchmark_submission(benchmark_submission_id)
        return submission

    async def create_project(
        self,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> CreateProjectCreateProject:
        input = CreateProject(name=name, description=description, tags=tags)
        return await self.client.create_project(input)

    def set_project(self, project_id: str):
        self.project_id = project_id

    async def run_benchmark(
        self,
        benchmark_id: str,
        rex_fn: str,
        name: str | None = None,
        sample: float | None = None,
        with_outs: bool | None = None,
    ) -> RunBenchmarkRunBenchmark:
        if not self.project_id:
            raise Exception("Please set up a project first with client.create_project and client.set_project")
        input = CreateRun(rex=rex_fn, name=name, project_id=self.project_id)
        res = await self.client.run_benchmark(input, benchmark_id, sample_pct=sample, with_outs=with_outs)
        if res.id:
            ui_url = UI_URL_MAP[self.client.url.strip("/")]
            print(
                f"View your submission at {ui_url}/project/{self.project_id}/runs?selectedRunId={res.source_run.id}"
            )
        return res

    async def eval_rex(
        self,
        rex_fn: str,
        name: str | None = None,
        wait_for_result: bool | None = None
    ) -> EvalEval:
        if not self.project_id:
            raise Exception("Please set up a project first with client.create_project and client.set_project")
        input = CreateRun(rex=rex_fn, name=name, project_id=self.project_id)
        res = await self.client.eval(input)
        if res.id:
            ui_url = UI_URL_MAP[self.client.url.strip("/")]
            print(
                f"View your submission at {ui_url}/project/{self.project_id}/runs?selectedRunId={res.id}"
            )
        if wait_for_result:
            return (await self.poll_run(res.id)).result
        return res

    async def poll_run(
        self,
        run_id: str,
    ) -> RunMeAccountProjectRun:
        run = await self.rex_run(run_id)
        while run.status not in ["DONE", "ERROR"]:
            await asyncio.sleep(5)
            run = await self.rex_run(run_id)
        return run

    # get a rex run
    async def rex_run(
        self,
        run_id: str,
    ) -> RunMeAccountProjectRun:
        return (await self.client.run(self.project_id, run_id)).account.project.run

    async def projects(
        self,
        after: str | None = None,
        before: str | None = None,
        first: int | None = None,
        last: int | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterable[Page[ProjectsMeAccountProjects, ProjectsMeAccountProjectsPageInfo]]:
        """
        Retrieve a list of projects.
        """

        filter = ProjectFilter(metadata=MetadataFilter(deleted_at=DateTimeFilter(is_null=True)))

        if tags:
            filter = ProjectFilter(
                all=[
                    filter,
                    ProjectFilter(metadata=MetadataFilter(tags=TagFilter(in_=tags))),
                ]
            )

        async def return_paged(
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[ProjectsMeAccountProjectsEdgesNode, ProjectsMeAccountProjectsPageInfo]:
            res = await self.client.projects(first=first, after=after, last=last, before=before, **kwargs)
            # The types for this pass in mypy, but not in pyright
            return res.account.projects  # type: ignore

        return self._query_with_pagination(
            return_paged,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {
                "filter": filter,
            },
        )

    async def object(self, path: UUID):
        """
        Retrieve an object from the database.

        :param id: The ID of the object.
        :return: The object.
        """
        # retry the download if it fails
        retries = 3
        while retries > 0:
            try:
                return await self.client.object_url(path)
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                else:
                    await asyncio.sleep(1)

    async def download_object(
        self,
        path: UUID,
        filename: str | None = None,
        filepath: Path | str | None = None,
        overwrite: bool = False,
        signed: bool = True,
        decode: bool = False,
    ):
        """
        Retrieve an object from the store: a wrapper for object with simpler behavior.

        :param id: The ID of the object.
        :param filepath: Where to download the object.
        :param filename: Download to the workspace with this name under "objects".
        """
        obj = (await self.object(path)) if signed else (await self.client.object_contents(path))
        if not obj:
            return None

        if filepath and isinstance(filepath, str):
            filepath = Path(filepath)

        if filepath is None:
            if filename is None:
                filename = str(path)
            if filename and self.workspace:
                if not (self.workspace / "objects").exists():
                    (self.workspace / "objects").mkdir()
                filepath = self.workspace / "objects" / filename

                if filepath.exists() and not overwrite:
                    # warn user that file is being restored
                    self.logger.warning(f"File {filename} already exists in workspace")
                    return filepath

        if filepath:
            if filepath.exists() and not overwrite:
                self.logger.warning(f"File {filename} already exists in workspace")
                return filepath
            if obj and isinstance(obj, ObjectContentsObjectPath):
                json.dump(obj.contents, open(filepath, "w"), indent=2)
            elif obj and obj.url:
                with httpx.stream(method="get", url=obj.url) as r:
                    r.raise_for_status()

                    buf = ""
                    with open(filepath, "wb") as f:
                        if not decode:
                            for chunk in r.iter_bytes():
                                f.write(chunk)
                            return filepath
                        first_chunk = True
                        is_encoded = False
                        for chunk in r.iter_text():
                            if not first_chunk and not is_encoded:
                                f.write(chunk.encode("utf-8"))
                                continue

                            # handle json
                            if first_chunk:
                                if len(chunk) > 0 and (chunk[0] == "[" or chunk[0] == "{"):
                                    f.write(chunk.encode("utf-8"))
                                    first_chunk = False
                                    continue
                                else:
                                    first_chunk = False
                                    is_encoded = True

                            # handle quotes
                            if len(chunk) > 0 and chunk[0] == '"':
                                chunk = chunk[1:]
                            if len(chunk) > 0 and chunk[-1] == '"':
                                chunk = chunk[:-1]
                            if len(chunk) == 0:
                                continue

                            len_to_take = math.floor(len(chunk) / 4) * 4
                            if (len(chunk) - len_to_take) >= (4 - len(buf)):
                                # if we have enough data to round out a multiple of 4
                                len_to_take += 4 - len(buf)
                            elif len_to_take - len(buf) > 0:
                                # if we can trim our amount to take to get a multiple of 4
                                len_to_take -= len(buf)
                            else:
                                # if we don't and can't
                                buf += chunk
                                continue

                            f.write(base64.b64decode(buf + chunk[:len_to_take]))
                            buf = chunk[len_to_take:]

            return filepath

        else:
            return obj

    def load_module_paths(self, filepath: Path) -> dict[str, str]:
        """
        Load all of the module versions from a file.

        :param filename: Json module version file
        """
        modules = None

        async def get_latest_modules(modules: dict[str, str]):
            async for page in await self.latest_modules():
                for edge in page.edges:
                    module = edge.node
                    if module.name in modules:
                        if module.path != modules[module.name]:
                            self.logger.warning(
                                f"""Module {module.name} has a different version on the server: {module.path}.
                                Use `.update_modules()` to update the lock file"""
                            )
                    else:
                        self.logger.warning(f"Module {module.path} is not in the lock file")

        if filepath.exists() and filepath.stat().st_size > 0:
            with open(filepath, "r") as f:
                modules = json.load(f)
            self.module_paths = modules

            # check against latest modules

            asyncio_run(get_latest_modules(modules), override="task")
            return modules
        else:
            raise FileNotFoundError("Lock file not found")

    def save_module_paths(self, modules: dict[str, str], filename: Path | None = None):
        """
        Lock all of the module versions to a file.

        :param modules: The modules to lock.
        """
        filename = filename or Path(f'modules-{time.strftime("%Y%m%dT%H%M%S")}.json')
        with open(filename, "w") as f:
            json.dump(modules, f, indent=2)
        return filename

    async def download_module_instance_output(
        self,
        instance_id: ModuleInstanceId,
        folder_path: Path | str | None = None,
        overwrite: bool = False,
    ):
        """
        Download an output from a module instance.

        :param instance_id: The ID of the module instance.
        :param output_id: The ID of the output.
        :param filename: Download to the workspace with this name under "objects".
        :param filepath: Where to download the object.
        """

        if folder_path and isinstance(folder_path, str):
            folder_path = Path(folder_path)
        else:
            folder_path = folder_path or (self.workspace or Path.cwd()) / f"outputs-{instance_id}"

        if folder_path.exists():
            if not overwrite:
                self.logger.warning(f"Folder {folder_path} already exists in workspace")
                return folder_path
        else:
            folder_path.mkdir(parents=True, exist_ok=overwrite)

        instance = await self.module_instance(instance_id)
        for obj in instance.outs:
            if obj.value and isinstance(obj.value, dict) and "path" in obj.value:
                await self.Arg(self, obj.id).download(filepath=folder_path / str(obj.id), overwrite=overwrite)

        return folder_path

    async def run(
        self,
        path: str,
        args: """list[
        BaseProvider.Arg[Any] | BaseProvider.BlockingArg[Any] | Argument | ArgId | Path | IOBase | Any
        ]""",
        target: Target | None = None,
        resources: Resources | None = None,
        tags: Sequence[str] | None = None,
        out_tags: Sequence[Sequence[str] | None] | None = None,
        restore: bool | None = None,
    ) -> RunModuleInstanceRunModuleInstance | ModuleInstanceFullModuleInstance:
        """
        Run a module with the given inputs and outputs.
        :param path: The path of the module.
        :param args: The arguments to the module.
        :param target: The target to run the module on.
        :param resources: The resources to run the module with.
        :param tags: The tags to apply to the module.
        :param out_tags: The tags to apply to the outputs of the module.
                         If provided, must be the same length as the number of outputs.
        :param restore: Check if a module instance with the same tags and path already exists.
        """
        tags = list(tags) + self.batch_tags if tags else self.batch_tags

        try_restore = restore if restore is not None else self.restore_by_default
        if try_restore:
            self.logger.info(f"Trying to restore job with tags: {tags} and path: {path}")
            res: list[ModuleInstanceFullModuleInstance] = []
            async for page in await self.module_instances(tags=list(tags), path=path):
                for edge in page.edges:
                    instance = edge.node
                    res.append(instance)
                    if len(res) > 1:
                        self.logger.warning("Multiple module instances found with the same tags and path")
            if len(res) >= 1:
                self.logger.info(f"Restoring job from previous run with id {res[0].id}")
                return res[0]

        # always request a bit of space because the run will always create files
        storage_requirements = {"storage": 10 * 1024 * 1024}

        # TODO: less insane version of this
        def gen_arg_dict(
            input: BaseProvider.Arg[Any] | BaseProvider.BlockingArg[Any] | ArgId | UUID | VirtualObject | Any,
        ) -> ArgumentInput:
            arg = ArgumentInput()
            if isinstance(input, BaseProvider.Arg) or isinstance(input, BaseProvider.BlockingArg):
                if input.id is None:
                    arg.value = input.value
                else:
                    arg.id = input.id
            elif isinstance(input, ArgId):
                arg.id = input
            else:
                arg = ArgumentInput(value=input)
            return arg

        arg_dicts = [gen_arg_dict(input) for input in args]

        if not resources:
            resources = ResourcesInput(
                storage=max(
                    int(math.ceil(storage_requirements["storage"] / 1024 / 1024)),
                    100,
                ),
                storage_units=MemUnits.MB,
            )

        lot = []
        if out_tags:
            lot = [list(t) if t else None for t in out_tags]

        if isinstance(target, str):
            target = ModuleInstanceTarget(target)
        if isinstance(resources, dict):
            resources = ResourcesInput(**resources)
        runres = await self.client.run_module_instance(
            instance=CreateModuleInstance(
                path=path,
                args=arg_dicts,
                target=target,
                resources=resources,
                out_tags=lot,
                metadata=CreateMetadata(tags=list(tags)),
            )
        )
        if not self.history:
            self.history = History(tags=list(tags), instances=[])
        self.history.instances.append(
            ModuleInstanceHistory(
                path=path,
                id=runres.id,
                status=ModuleInstanceStatus.CREATED,
                tags=tags,
            )
        )
        if self.workspace:
            self.save()
        return runres

    async def module_instances(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        path: Optional[str] = None,
        status: Optional[ModuleInstanceStatus] = None,
        tags: Optional[Sequence[str]] = None,
        ids: Optional[list[ModuleInstanceId]] = None,
    ) -> AsyncIterable[Page[ModuleInstanceFullModuleInstance, PageInfoFull]]:
        """
        Retrieve a list of module instancees filtered by the given parameters.

        :param first: Retrieve the first N module instances.
        :param after: Retrieve module instances after a certain cursor.
        :param last: Retrieve the last N module instances.
        :param before: Retrieve module instances before a certain cursor.
        :param path: Retrieve module instancees with for the given module path.
        :param status: Retrieve module instancees with the specified status (CREATED, RUNNING, etc.).
        :param tags: Retrieve module instancees with the given list of tags.
        :return: A list of filtered module instancee.
        """

        async def return_paged(
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[ModuleInstancesMeAccountModuleInstancesEdgesNode, PageInfoFull]:
            res = await self.client.module_instances(
                first=first, after=after, last=last, before=before, **kwargs
            )
            # FIXME: this passes in mypy but not in pyright
            return res.account.module_instances  # type: ignore

        # construct filter like
        # filters: list[ModuleFilter] = []
        # tag_filter = TagFilter()
        # if isinstance(path, str):
        #     filters.append(ModuleFilter(path=StringFilter(eq = path)))
        # if isinstance(tags, list):
        #     tag_filter.in_ = tags
        #     filters.append(ModuleFilter(metadata = MetadataFilter(tags=tag_filter)))

        # module_filter = ModuleFilter(all=filters) if filters else None
        filters: list[ModuleInstanceFilter] = []
        if isinstance(path, str):
            filters.append(ModuleInstanceFilter(path=StringFilter(eq=path)))
        if isinstance(status, ModuleInstanceStatus):
            filters.append(ModuleInstanceFilter(status=ModuleInstanceStatusFilter(eq=status)))
        if isinstance(tags, list):
            tag_filter = TagFilter()
            tag_filter.in_ = tags
            filters.append(ModuleInstanceFilter(metadata=MetadataFilter(tags=tag_filter)))
        if isinstance(ids, list):
            for id in ids:
                filters.append(ModuleInstanceFilter(id=UuidFilter(eq=id)))

        module_instance_filter = ModuleInstanceFilter(all=filters) if filters else None

        return self._query_with_pagination(
            return_paged,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {"filter": module_instance_filter},
            # {"path": path, "name": name, "status": status, "tags": tags, "ids": ids},
        )

    async def modules(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None,
        path: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> AsyncIterable[Page[ModuleFull, ModulesModulesPageInfo]]:
        """
        Get all modules.

        :param first: Retrieve the first N modules.
        :param after: Retrieve modules after a certain cursor.
        :param last: Retrieve the last N modules.
        :param before: Retrieve modules before a certain cursor.
        :param path: Retrieve modules with for the given module path.
        :param name: Retrieve modules with for the given module name.
        :param tags: Retrieve modules with the given list of tags.
        """

        filters: list[ModuleFilter] = []
        tag_filter = TagFilter()
        if isinstance(path, str):
            filters.append(ModuleFilter(path=StringFilter(eq=path)))
        if isinstance(tags, list):
            tag_filter.in_ = tags
            filters.append(ModuleFilter(metadata=MetadataFilter(tags=tag_filter)))

        module_filter = ModuleFilter(all=filters) if filters else None

        return self._query_with_pagination(
            self.client.modules,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {
                "module_filter": module_filter,
            },
        )

    async def latest_modules(
        self,
        first: Union[Optional[int], UnsetType] = UNSET,
        after: Union[Optional[str], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        names: Union[Optional[list[str]], UnsetType] = UNSET,
    ) -> AsyncIterable[Page[ModuleFull, LatestModulesLatestModulesPageInfo]]:
        """
        Get latest modules.
        """

        return self._query_with_pagination(
            self.client.latest_modules,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {
                "names": names,
            },
        )

    async def runs(
        self,
    ) -> AsyncIterable[Page[RunsMeAccountProjectRunsEdgesNode, RunsMeAccountProjectRunsPageInfo]]:
        """
        Retrieve a list of runs.
        """
        if not self.project_id:
            raise Exception("No project ID provided")

        async def return_paged(
            project_id: str,
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[RunsMeAccountProjectRunsEdgesNode, RunsMeAccountProjectRunsPageInfo]:
            res = await self.client.runs(
                project_id=project_id,
                first=first,
                after=after,
                last=last,
                before=before,
                **kwargs,
            )
            # The types for this pass in mypy, but not in pyright
            return res.account.project.runs  # type: ignore

        return self._project_query_with_pagination(return_paged, UUID(self.project_id), PageVars(), {})

    async def get_latest_module_paths(self, names: list[str] | None = None) -> dict[str, str]:
        """
        Get the latest module paths for a list of modules.

        :param names: The names of the modules.
        """
        ret = {}
        module_pages = [i async for i in await self.latest_modules(names=names)]
        for module_page in module_pages:
            for edge in module_page.edges:
                path = edge.node.path
                name = get_name_from_path(edge.node.path)
                if path:
                    ret[name] = path
        return ret

    async def get_modules_for_paths(
        self, paths: list[str]
    ) -> AsyncIterable[Page[ModuleFull, ModulesModulesPageInfo]]:
        """
        Get modules for the provided paths.
        """
        for path in paths:
            ms = await self.modules(path=path)
            mps = [m async for m in ms]
            if len(mps) != 1:
                self.logger.warning(f"Found no modules for path {path} - remove your lockfile and try again")
            else:
                yield mps[0]

    async def tag(
        self,
        tags: list[str],
        module_id: str | None = None,
        module_instance_id: str | None = None,
        argument_id: str | None = None,
    ) -> list[str]:
        """
        Add a list of tags to a module, module instance, or argument.

        :param tags: The list of tags to be added.
        :param module_id: The ID of the module to be tagged.
        :param module_instance_id: The ID of the module instance to be tagged.
        :param argument_id: The ID of the argument to be tagged.
        :return: The resulting full list of tags on the entity.
        """
        return await self.client.tag_module(
            tags=tags,
            moduleId=module_id,
            moduleInstanceId=module_instance_id,
            argumentId=argument_id,
        )

    async def untag(
        self,
        tags: list[str],
        module_id: str | None = None,
        module_instance_id: str | None = None,
        argument_id: str | None = None,
    ) -> list[str]:
        """
        Remove a list of tags from a module, module instance, or argument.

        :param tags: The list of tags to be removed.
        :param module_id: The ID of the module to be untagged.
        :param module_instance_id: The ID of the module instance to be untagged.
        :param argument_id: The ID of the argument to be untagged.
        :return: The list of remaining tags.
        """
        return await self.client.untag_module(
            tags=tags,
            moduleId=module_id,
            moduleInstanceId=module_instance_id,
            argumentId=argument_id,
        )

    async def get_module_functions(
        self,
        names: list[str] | None = None,
        tags: list[str] | None = None,
        lockfile: Path | None = None,
    ) -> dict[str, RushModuleRunner[Any]]:
        """
        Get a dictionary of module functions.

        :param names: The names of the modules.
        :param lockfile: The lockfile to load the modules from.
        :return: A dictionary of module functions.
        """

        ret: dict[str, RushModuleRunner[Any]] = {}
        if lockfile is not None:
            module_paths = self.load_module_paths(lockfile)
            module_pages = self.get_modules_for_paths(list(module_paths.values()))
        else:
            if self._config_dir:
                if self.module_paths.items():
                    # we have already loaded a lock via the workspace
                    module_pages = self.get_modules_for_paths(list(self.module_paths.values()))
                else:
                    # lets load the latest paths and lock them
                    if tags:
                        module_pages = await self.modules(tags=tags)
                        self.module_paths = {}
                        async for module_page in module_pages:
                            for edge in module_page.edges:
                                path = edge.node.path
                                name = get_name_from_path(edge.node.path)
                                if (names and name in names) and path:
                                    self.module_paths[name] = path
                        module_pages = await self.modules(tags=tags)
                    else:
                        paths = await self.get_latest_module_paths(names)
                        module_pages = self.get_modules_for_paths(list(paths.values()))
                        self.module_paths = paths
                        self.save_module_paths(self.module_paths, self._config_dir / "rush.lock")
            elif tags:
                # no workspace, so up the user to lock it
                module_pages = await self.modules(tags=tags)
            else:
                # no workspace, so up the user to lock it
                module_pages = await self.latest_modules(names=names)

        # so that our modules get constructed in sorted order for docs
        modules: dict[str, Any] = {}
        async for module_page in module_pages:
            for edge in module_page.edges:
                module = edge.node.__deepcopy__()
                path = module.path
                name = get_name_from_path(edge.node.path)
                # in the case of if not self.config dir and names and tags,
                # we have to filter by the names still, so do it here
                if names and name not in names and tags:
                    continue
                modules[name] = module

        for name in sorted(modules.keys()):
            module = modules[name]
            path = module.path

            out_types = tuple(type_from_typedef(i) for i in module.outs)

            def random_target():
                allowed_default_targets = ["BULLET", "BULLET_2"]
                if "BULLET_3" in str(module.targets) or "BULLET_3_GPU" in str(module.targets):
                    allowed_default_targets.append("BULLET_3")
                return random.choice(allowed_default_targets)

            default_resources = None
            if module.resource_bounds:
                default_resources = {
                    "storage": (module.resource_bounds.storage_min or 0) + 10,
                    "storage_units": "MB",
                    "gpus": module.resource_bounds.gpu_hint or 0,
                }

            def closure(
                name: str,
                path: str,
                module_ins: list[Any],
                module_outs: list[Any],
                default_resources: dict[str, Any] | None,
            ) -> Callable[
                ...,
                Coroutine[
                    Any,
                    Any,
                    tuple[BaseProvider.Arg[Any] | BaseProvider.BlockingArg[Any], ...],
                ],
            ]:
                in_types = tuple(type_from_typedef(i) for i in module_ins)
                typechecker = build_typechecker(*in_types)

                async def runner(
                    *args: Any,
                    target: Target | None = None,
                    resources: Resources | None = default_resources,
                    tags: list[str] | None = None,
                    output_tags: Sequence[Sequence[str] | None] | None = None,
                    restore: bool | None = None,
                ) -> tuple[BaseProvider.Arg[Any] | BaseProvider.BlockingArg[Any], ...]:
                    if not output_tags and tags:
                        output_tags = [tags] * len(module_outs)
                    args = await self.upload_args(args, module_ins)
                    if target is None:
                        target = random_target()
                    typechecker(*args)
                    run = await self.run(
                        path,
                        list(args),
                        target,
                        resources,
                        tags,
                        out_tags=output_tags,
                        restore=restore,
                    )
                    return tuple(
                        (BaseProvider.BlockingArg if self.__is_blocking__ else BaseProvider.Arg)(
                            self, out.id, source=run.id
                        )
                        for out in run.outs
                    )

                runner.__name__ = name

                # convert ins_usage array to argument docs
                ins_docs = ""
                if module.ins_usage:
                    for ins in module.ins_usage:
                        # replace the first non-markup colon on the first line with a semicolon,
                        # so the docs don't get rendered incorrectly
                        ins_firstline, ins_rest = (
                            ins.split("\n")[0],
                            "\n".join(ins.split("\n")[1:]),
                        )
                        ins_parts = ins_firstline.split(":")
                        if len(ins_parts) > 2:
                            ins = (
                                ins_parts[0]
                                + ":"
                                + ins_parts[1]
                                + ";"
                                + ":".join(ins_parts[2:])
                                + ("\n" if ins_rest else "")
                                + ins_rest
                            )
                        ins_docs += f"\n:param {ins}"

                # convert outs_usage array to return docs
                outs_docs = ""
                if module.outs_usage:
                    for outs in module.outs_usage:
                        outs_docs += f"\n:return {outs}"

                if module.description:
                    runner.__doc__ = (
                        module.description
                        + "\n\nModule version:  \n`"
                        + path
                        + "`\n\nQDX Type Description:\n\n"
                        + format_module_typedesc(module.typedesc)
                        + "\n"
                        + (ins_docs)
                        + (outs_docs)
                    )
                else:
                    runner.__doc__ = name + " @" + path

                if sys.version_info >= (3, 11):
                    exec(
                        "runner.__annotations__['args'] = (*tuple[*(t.to_python_type() for t in in_types)],)[0]"  # noqa: E501
                    )
                    exec("runner.__annotations__['return'] = tuple[*(t.to_python_type() for t in out_types)]")
                else:
                    from typing_extensions import Unpack

                    runner.__annotations__["args"] = Unpack[
                        tuple[tuple(t.to_python_type() for t in in_types)]
                    ]
                    runner.__annotations__["return"] = tuple[tuple(t.to_python_type() for t in out_types)]

                return runner

            runner = closure(name, path, module.ins, module.outs, default_resources)
            self.__setattr__(name, runner)
            ret[name] = runner
        return ret

    async def upload_args(
        self,
        args: tuple[Any, ...],
        in_types: list[Any],
    ) -> tuple[Any, ...]:
        """
        Walk through input types and for any that are files, upload them.
        Replace the file with the uploaded object in the arg list and return the list.

        :param args: The arguments to be uploaded.
        :param in_types: The types of the arguments.

        :return: Arguments with files replaced with virtual objects.
        """
        newargs: list[Any] = []
        for i, arg in enumerate(args):
            if isinstance(arg, Path):
                obj = await self.upload(arg, in_types[i])
                newargs.append(obj.object)
            else:
                # if arg is a record, we need to check if any members are a path
                if isinstance(arg, dict):
                    for key, value in arg.items():
                        if isinstance(value, Path):
                            obj = await self.upload(value, in_types[i].members[key])
                            arg[key] = obj.object
                if isinstance(arg, list):
                    arg_list: list[Any] = arg
                    for j, value in enumerate(arg_list):
                        if isinstance(value, Path):
                            obj = await self.upload(value, in_types[i]["t"])
                            arg[j] = obj.object
                newargs.append(arg)
        return tuple(newargs)

    async def retry(
        self,
        id: ModuleInstanceId,
        target: Target,
        resources: Resources | None = None,
    ) -> RetryRetry:
        """
        Retry a module instance.

        :param id: The ID of the module instance to be retried.
        :return: The ID of the new module instance.
        """
        return await self.client.retry(instance=id, resources=resources, target=target)  # type: ignore

    async def upload(
        self,
        file: Path | str,
        typeinfo: dict[str, Any] | RushType[Any],
    ):
        """
        Upload an Object with typeinfo and store as an Argument.

        :param file: The file to be uploaded.
        :param typeinfo: The typeinfo of the file.
        """
        if isinstance(file, str):
            file = Path(file)
        with open(file, "rb") as f:
            format = ObjectFormat.json if file.suffix == ".json" else ObjectFormat.bin
            # TODO: see if we can use this in the upload
            # mimetype = mimetypes.guess_type(file)[0]
            meta = await self.client.upload_large_object(
                typeinfo=typeinfo,
                format=format,
            )
            # use upload url to PUT file to
            httpx.put(
                meta.upload_url,
                content=f.read(),
                headers={"content-type": "application/octet-stream"},
                timeout=httpx.Timeout(600),
            )
            return meta.descriptor

    async def module_instance(self, id: ModuleInstanceId) -> ModuleInstanceDetailsModuleInstance:
        """
        Retrieve a module instance by its ID.

        :param id: The ID of the module instance to be retrieved.
        :return: The retrieved module instance.
        :raises Exception: If the module instance is not found.
        """
        return await self.client.module_instance_details(id)

    async def logs(
        self,
        id: ModuleInstanceId,
        kind: Literal["stdout", "stderr"],
        after: str | None = None,
        before: str | None = None,
        pages: int | None = None,
    ) -> AsyncIterable[str]:
        """
        Retrieve the stdout and stderr of a module instance.

        :param id: The ID of the module instance.
        :return: The stdout and stderr of the module instance.
        """

        # page through the logs

        async def return_paged(
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            _: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[Any, Any]:
            args = {}

            if kind == "stdout":
                args = {
                    "stdout_after": after,
                    "stdout_before": before,
                }
            else:
                args = {
                    "stderr_after": after,
                    "stderr_before": before,
                }

            res = await self.client.module_instance_details(id, **args)

            return res.stderr if kind == "stderr" else res.stdout  # type: ignore

        i = 0
        async for page in self._query_with_pagination(
            return_paged,  # type: ignore
            PageVars(after=after, before=before),
            {},
        ):
            for edge in page.edges:
                yield edge.node.content
                i += 1
                if pages is not None and i > pages:
                    return

    async def update_modules(self, names: list[str] | None = None, tags: list[str] | None = None):
        """
        Update the module paths in the lockfile.

        :param names: Optional list of names to update.
        :param tags: Optionally only upate modules with this tag.
        """
        if not self._config_dir:
            raise Exception("No workspace provided")

        if tags:
            module_pages = await self.modules(tags=tags)
            self.module_paths = {}
            async for module_page in module_pages:
                for edge in module_page.edges:
                    path = edge.node.path
                    name = get_name_from_path(edge.node.path)
                    if names:
                        if name in names and path:
                            self.module_paths[name] = path
                    else:
                        self.module_paths[name] = path
            self.save_module_paths(self.module_paths, self._config_dir / "rush.lock")
        else:
            paths = await self.get_latest_module_paths(names)
            self.module_paths = paths
            self.save_module_paths(self.module_paths, self._config_dir / "rush.lock")

        built_fns = await self.get_module_functions(names=names, tags=tags)

        if self.__is_blocking__:
            _make_blocking(self, built_fns)

    async def delete_module_instance(self, id: ModuleInstanceId):
        """
        Delete a module instance with a given ID.

        :param id: The ID of the module instance to be deleted.
        :return: The ID of the deleted module instance.
        :raises RuntimeError: If the operation fails.
        """
        return await self.client.delete_module_instance(id)

    async def cancel(self, id: ModuleInstanceId):
        """
        Cancel a module instance.

        :param id: The ID of the module instance to be cancelled.
        :return: The ID of the cancelled module instance.
        """
        return await self.client.cancel_module_instance(id)

    async def poll_module_instance(
        self, id: ModuleInstanceId, n_retries: int = 10, poll_rate: int = 30
    ) -> Any:
        """
        Poll a module instance until it is completed, with a specified number of retries and poll rate.

        We do exponential backoff from 0.5s, so a retry only counts once we hit the input poll rate.

        :param id: The ID of the module instance to be polled.
        :param n_retries: The maximum number of retries. Default is 10.
        :param poll_rate: The poll rate in seconds. Default is 30.
        :return: The completed module instance.
        :raises Exception: If the module instance fails or polling times out.
        """
        n_try = 0

        curr_poll_rate = 0.5
        while n_try < n_retries:
            await asyncio.sleep(curr_poll_rate)
            if curr_poll_rate == poll_rate:
                n_try += 1
            curr_poll_rate = min(curr_poll_rate * 2, poll_rate)
            module_instance = await self.client.module_instance_minimal(id=id)
            if module_instance and module_instance.status in ["COMPLETED", "FAILED"]:
                return module_instance

        raise Exception("Module polling timed out")

    async def set_default_project(self):
        if not self.project_id:
            p = await anext((await self.projects(tags=["default"])))
            if p and len(p.edges) > 0:
                self.project_id = p.edges[0].node.id
            else:
                self.project_id = (await self.create_project("Default Project", tags=["default"])).id

    class Arg(Generic[T]):
        def __init__(
            self,
            provider: "BaseProvider | None",
            id: UUID | None = None,
            source: UUID | None = None,
            value: T | None = None,
            typeinfo: dict[str, Any] | SCALARS | None = None,
        ):
            self.provider = provider
            self.id = id
            self.source = source
            self.status = ModuleInstanceStatus.CREATED
            self.progress = ModuleInstanceFullProgress(n=0, n_max=0, n_expected=0, done=False)
            self.value = value
            self.typeinfo = typeinfo

        def __repr__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __str__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.id == other.id

        async def info(self) -> ArgumentArgument:
            return await self._info()

        async def _info(self) -> ArgumentArgument:
            async def get_remote_arg(retries: int):
                if self.id is None:
                    raise Exception("No ID provided")
                if self.provider is None:
                    raise Exception("No provider provided")
                try:
                    remote_arg = await self.provider.argument(self.id)
                    self.typeinfo = remote_arg.typeinfo
                    return remote_arg
                except GraphQLClientGraphQLMultiError as e:
                    if e.errors and e.errors[0].message == "not found":
                        if retries > 0:
                            if self.source:
                                module_instance = await self.provider.module_instance(self.source)
                                if module_instance.status != self.status:
                                    self.provider.logger.info(
                                        f"Argument {self.id} is now {module_instance.status}"
                                    )
                                    self.status = module_instance.status
                                if module_instance.status == ModuleInstanceStatus.RUNNING:
                                    if (
                                        module_instance.progress
                                        and module_instance.progress.n != self.progress.n
                                    ):
                                        print(
                                            f"Progress: {module_instance.progress}",
                                            end="\r",
                                        )
                                    else:
                                        print(
                                            "module running with no progress reported",
                                            end="\r",
                                        )
                            await asyncio.sleep(5)
                        else:
                            raise e
                    else:
                        self.provider.logger.error(e.errors)
                        raise e

            retries = 10
            remote_arg = await get_remote_arg(retries)
            while remote_arg is None:
                retries -= 1
                remote_arg = await get_remote_arg(retries)
            return remote_arg

        async def download(
            self,
            filename: str | None = None,
            filepath: Path | None = None,
            overwrite: bool = False,
        ):
            return await self._download(filename, filepath, overwrite)

        async def _download(
            self,
            filename: str | None = None,
            filepath: Path | None = None,
            overwrite: bool = False,
        ):
            if self.id is None:
                raise Exception("No ID provided")
            if self.provider is None:
                raise Exception("No provider provided")
            await self._get()

            if self.typeinfo:
                if isinstance(self.typeinfo, dict) and (
                    (self.typeinfo.get("k") == "record" and self.typeinfo.get("n") == "Object")
                    or (
                        self.typeinfo.get("k") == "optional"
                        and isinstance(self.typeinfo.get("t"), dict)
                        and self.typeinfo["t"].get("k") == "record"
                        and self.typeinfo["t"].get("n") == "Object"
                    )
                    or (
                        self.typeinfo.get("k") == "fallible"
                        and isinstance(self.typeinfo.get("t"), list)
                        and self.typeinfo["t"][0].get("k") == "record"
                        and self.typeinfo["t"][0].get("n") == "Object"
                    )
                ):
                    signed = "$" in json.dumps(self.typeinfo)
                    decode = not (isinstance(self.value, dict) and self.value.get("format") == "bin")
                    if isinstance(self.value, dict) and "Ok" in self.value:
                        self.value = self.value["Ok"]
                    if isinstance(self.value, dict) and "path" in self.value:
                        path: str = self.value["path"] if isinstance(self.value["path"], str) else ""
                        return await self.provider.download_object(
                            UUID(hex=path),
                            filename,
                            filepath,
                            overwrite,
                            signed,
                            decode,
                        )
                    else:
                        raise Exception("Invalid value format for object argument")
                else:
                    raise Exception("Cannot download non-object argument")
            else:
                raise Exception("Cannot download argument without typeinfo")

        async def get(self) -> T | str:
            return await self._get()

        async def _get(self) -> T | str:
            """
            Get the value of the argument.

            This will wait until the argument is ready.

            :return: The value of the argument.
            """
            if self.value is None or self.typeinfo is None:
                if self.provider is None:
                    raise Exception("No provider provided")
                if self.id is None:
                    raise Exception("No ID provided")
                while self.value is None:
                    try:
                        remote_arg = await self._info()
                        if remote_arg.rejected_at:
                            if remote_arg.source:
                                # get the failure reason by checking the source module instance
                                module_instance = await self.provider.module_instance(remote_arg.source)
                                raise Exception(
                                    (
                                        getattr(module_instance, "failure_reason", None),
                                        getattr(module_instance, "failure_context", None),
                                    )
                                )
                            raise Exception("Argument was rejected")
                        else:
                            self.value = remote_arg.value
                            if self.value is None:
                                source = remote_arg.source or self.source
                                if source:
                                    module_instance = await self.provider.module_instance(source)
                                    if module_instance.status != self.status:
                                        self.provider.logger.info(
                                            f"Argument {self.id} is now {module_instance.status}"
                                        )
                                        self.status = module_instance.status

                                    if module_instance.status == ModuleInstanceStatus.RUNNING:
                                        if (
                                            module_instance.progress
                                            and module_instance.progress.n != self.progress.n
                                        ):
                                            print(
                                                f"Progress: {module_instance.progress}",
                                                end="\r",
                                            )
                                        else:
                                            print(
                                                "module running with no progress reported",
                                                end="\r",
                                            )
                                await asyncio.sleep(1)
                    except GraphQLClientGraphQLMultiError as e:
                        if e.errors and e.errors[0].message == "not found":
                            await asyncio.sleep(1)
                        else:
                            self.provider.logger.error(e.errors)
                            raise e

            # if typeinfo is a dict, check if it is an object, and if so, download it
            if isinstance(self.typeinfo, dict) and self.provider and self.id:
                if (self.typeinfo.get("k") == "record" and self.typeinfo.get("n") == "Object") or (
                    self.typeinfo.get("k") == "optional"
                    and isinstance(self.typeinfo.get("t"), dict)
                    and self.typeinfo["t"].get("k") == "record"
                    and self.typeinfo["t"].get("n") == "Object"
                ):
                    if (
                        isinstance(self.value, dict)
                        and "path" in self.value
                        and isinstance(self.value["path"], str)
                    ):
                        obj = await self.provider.object(UUID(self.value["path"]))
                        if obj and obj.url:
                            return obj.url
                        raise Exception("Object not found")
            return cast(T, self.value)

    class BlockingArg(Arg[T]):
        def __init__(
            self,
            provider: "BaseProvider | None",
            id: UUID | None = None,
            source: UUID | None = None,
            value: T | None = None,
            typeinfo: dict[str, Any] | SCALARS | None = None,
        ):
            super().__init__(provider, id, source, value, typeinfo)

        def get(self) -> T | str:
            return asyncio_run(super().get())

        def download(
            self,
            filename: str | None = None,
            filepath: Path | None = None,
            overwrite: bool = False,
        ):
            return asyncio_run(super().download(filename, filepath, overwrite))

        def info(self) -> ArgumentArgument:
            return asyncio_run(super().info())


class Provider(BaseProvider):
    def __init__(
        self,
        access_token: str | None = None,
        url: str | None = None,
        workspace: str | Path | bool | None = None,
        project: UUID | None = None,
        batch_tags: list[str] | None = None,
        logger: logging.Logger | None = None,
        restore_by_default: bool | None = None,
    ):
        """
        Initialize the RushProvider with a graphql client.

        :param access_token: The access token to use.
        :param url: The url to use.
        :param workspace: The workspace directory to use.
        :param batch_tags: The tags that will be placed on all runs by default.
        """
        if workspace is None:
            workspace = Path(".")
        if workspace is True:
            workspace = Path(".")
        if workspace is False:
            workspace = None

        if not logger:
            logger = logging.getLogger("rush")
            if len(logger.handlers) == 0:
                stderr_handler = logging.StreamHandler()
                stderr_handler.setLevel(logging.ERROR)
                stderr_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )

                stdout_handler = logging.StreamHandler(sys.stdout)
                stdout_handler.setLevel(logging.INFO)
                stdout_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )

                # add filter to prevent errors from being logged twice
                stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
                logger.setLevel(logging.INFO)

                logger.addHandler(stdout_handler)
                logger.addHandler(stderr_handler)

        if os.getenv("RUSH_RESTORE_BY_DEFAULT") == "True" and restore_by_default is None:
            logger.info("Restoring by default via env")
            restore_by_default = True
        elif os.getenv("RUSH_RESTORE_BY_DEFAULT") == "False" and restore_by_default is None:
            logger.info("Not restoring by default via env")
            restore_by_default = False

        elif restore_by_default is None:
            restore_by_default = False

        if access_token is None or url is None:
            # try to check the environment variables

            if access_token is None:
                access_token = os.environ.get("RUSH_TOKEN")
                if access_token is None:
                    raise Exception("No access token provided")

            if url is None:
                url = os.environ.get("RUSH_URL") or "https://tengu.qdx.ai/"
            client = Client(url=url, headers={"Authorization": f"bearer {access_token}"})

            super().__init__(
                client,
                workspace=workspace,
                batch_tags=batch_tags,
                logger=logger,
                project_id=project,
                restore_by_default=restore_by_default,
            )
        else:
            client = Client(url=url, headers={"Authorization": f"bearer {access_token}"})
            super().__init__(
                client,
                workspace=workspace,
                batch_tags=batch_tags,
                logger=logger,
                project_id=project,
                restore_by_default=restore_by_default,
            )


async def build_provider_with_functions(
    workspace: str | Path | bool | None = None,
    project: str | None = None,
    access_token: str | None = None,
    url: str | None = None,
    batch_tags: list[str] | None = None,
    module_names: list[str] | None = None,
    module_tags: list[str] | None = None,
    logger: logging.Logger | None = None,
    restore_by_default: bool | None = None,
) -> Provider:
    """
    Build a RushProvider with the given access token and url.

    :param access_token: The access token to use.
    :param url: The url to use.
    :param workspace: The workspace directory to use.
    :param batch_tags: The tags that will be placed on all runs by default.
    :return: The built RushProvider.
    """
    provider = Provider(
        access_token,
        url,
        workspace,
        UUID(project) if project else None,
        batch_tags,
        logger,
        restore_by_default=restore_by_default,
    )

    await provider.get_module_functions(names=module_names, tags=module_tags)
    # if the user didn't  specify a project, use or create the default project
    await provider.set_default_project()

    return provider


TA = TypeVar("TA")


async def wrap_coroutine(coro: Awaitable[TA]) -> TA:
    return await coro


def _make_blocking(provider: BaseProvider, built_fns: dict[str, RushModuleRunner[Any]]):
    # functions that don't get called internally can be overridden with blocking versions
    blockable_functions = (
        "nuke",
        "status",
        "logs",
        "retry",
        "tag",
        "update_modules",
        "download_module_instance_output",
        "run_benchmark",
        "benchmark",
        "eval_rex"
    )
    # for each async function in the provider, create a blocking version
    blocking_versions: dict[str, Callable[..., Any]] = {}
    for name, func in provider.__dict__.items():
        if (
            asyncio.iscoroutinefunction(func)
            or inspect.iscoroutine(func)
            or inspect.iscoroutinefunction(func)
        ):

            def closurea(func: Callable[..., Awaitable[TA]], name: str) -> Callable[..., TA]:
                def blocking_func(
                    *args: Any,
                    target: Target | None = None,
                    resources: Resources | None = None,
                    tags: list[str] | None = None,
                    restore: bool | None = None,
                ) -> Any:
                    return asyncio_run(
                        wrap_coroutine(
                            func(
                                *args,
                                target=target,
                                resources=resources,
                                tags=tags,
                                restore=restore,
                            )
                        )
                    )

                return blocking_func

            name = name if name in built_fns else f"{name}_blocking"
            blocking_func = closurea(func, name)
            blocking_func.__name__ = f"{name}"
            blocking_func.__doc__ = func.__doc__
            blocking_func.__annotations__ = func.__annotations__

            blocking_versions[name] = blocking_func

    for name, func in BaseProvider.__dict__.items():
        if (
            asyncio.iscoroutinefunction(func)
            or inspect.iscoroutine(func)
            or inspect.iscoroutinefunction(func)
        ):

            def closureb(func: Callable[..., Awaitable[Any]], name: str):
                def blocking_func(*args: Any, **kwargs: Any) -> Any:
                    r = asyncio_run(wrap_coroutine(func(provider, *args, **kwargs)))
                    if isinstance(r, AsyncGenerator):
                        r_a: AsyncGenerator[Any, Any] = r
                        res = []
                        while True:
                            try:
                                res.append(asyncio_run(wrap_coroutine(anext(r_a))))
                            except StopAsyncIteration:
                                return res
                    return r

                return blocking_func

            name = name if name in blockable_functions else f"{name}_blocking"
            blocking_func = closureb(func, name)
            blocking_func.__name__ = f"{name}"
            blocking_func.__doc__ = func.__doc__
            blocking_func.__annotations__ = func.__annotations__

            blocking_versions[name] = blocking_func

    for name, func in BaseProvider.__dict__.items():
        if inspect.isasyncgenfunction(func) or inspect.isasyncgen(func):

            def closurec(func: Callable[..., AsyncGenerator[Any, None]], name: str):
                def blocking_func(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
                    r = func(provider, *args, **kwargs)
                    try:
                        hn = asyncio_run(anext(r))
                    except StopAsyncIteration:
                        return
                    while True:
                        yield hn
                        try:
                            hn = asyncio_run(anext(r))
                        except StopAsyncIteration:
                            return
                    return r

                return blocking_func

            name = name if name in blockable_functions else f"{name}_blocking"
            blocking_func = closurec(func, name)
            blocking_func.__name__ = f"{name}"
            blocking_func.__doc__ = func.__doc__
            blocking_func.__annotations__ = func.__annotations__

            blocking_versions[name] = blocking_func

    provider.__dict__.update(blocking_versions)
    provider.__is_blocking__ = True


def build_blocking_provider_with_functions(
    workspace: str | Path | bool | None = None,
    access_token: str | None = None,
    url: str | None = None,
    batch_tags: list[str] | None = None,
    project: str | None = None,
    module_names: list[str] | None = None,
    module_tags: list[str] | None = None,
    logger: logging.Logger | None = None,
    restore_by_default: bool | None = None,
) -> Provider:
    """
    Build a RushProvider with the given access token and url.

    :param access_token: The access token to use.
    :param url: The url to use.
    :param workspace: The workspace directory to use.
    :param batch_tags: The tags that will be placed on all runs by default.
    :return: The built RushProvider.
    """
    provider = Provider(
        access_token,
        url,
        workspace,
        UUID(project) if project else None,
        batch_tags,
        logger,
        restore_by_default=restore_by_default,
    )
    if not LOOP.is_running():
        try:
            event_loop_running = asyncio.get_event_loop().is_running()
        except RuntimeError:
            event_loop_running = False

        if not event_loop_running:
            _LOOP_THREAD = threading.Thread(target=start_background_loop, args=(LOOP,), daemon=True)
            _LOOP_THREAD.start()

    built_fns = asyncio_run(provider.get_module_functions(names=module_names, tags=module_tags))

    _make_blocking(provider, built_fns)

    asyncio_run(provider.set_default_project())

    return provider

def build_blocking_provider(
    workspace: str | Path | bool | None = None,
    access_token: str | None = None,
    url: str | None = None,
    batch_tags: list[str] | None = None,
    project: str | None = None,
    module_names: list[str] | None = None,
    module_tags: list[str] | None = None,
    logger: logging.Logger | None = None,
    restore_by_default: bool | None = None,
) -> Provider:
    """
    Build a RushProvider with the given access token and url.

    :param access_token: The access token to use.
    :param url: The url to use.
    :param workspace: The workspace directory to use.
    :param batch_tags: The tags that will be placed on all runs by default.
    :return: The built RushProvider.
    """
    provider = Provider(
        access_token,
        url,
        workspace,
        UUID(project) if project else None,
        batch_tags,
        logger,
        restore_by_default=restore_by_default,
    )
    if not LOOP.is_running():
        try:
            event_loop_running = asyncio.get_event_loop().is_running()
        except RuntimeError:
            event_loop_running = False

        if not event_loop_running:
            _LOOP_THREAD = threading.Thread(target=start_background_loop, args=(LOOP,), daemon=True)
            _LOOP_THREAD.start()

    asyncio_run(provider.set_default_project())

    _make_blocking(provider, {})

    return provider
