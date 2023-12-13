import base64
import json
import math
import time
from dataclasses import dataclass
from io import IOBase
from pathlib import Path
from typing import Any, AsyncIterable, Generic, Iterable, List, Literal, Optional, Protocol, TypeVar, Union
from uuid import UUID
from collections import Counter

import httpx
from pydantic_core import to_jsonable_python

from .graphql_client.argument import Argument, ArgumentArgument
from .graphql_client.arguments import (
    ArgumentsMeAccountArguments,
    ArgumentsMeAccountArgumentsEdgesNode,
    ArgumentsMeAccountArgumentsPageInfo,
)
from .graphql_client.base_model import UNSET, UnsetType, Upload
from .graphql_client.client import Client
from .graphql_client.enums import MemUnits, ModuleInstanceStatus, ModuleInstanceTarget
from .graphql_client.exceptions import GraphQLClientGraphQLMultiError
from .graphql_client.fragments import ModuleFull, PageInfoFull
from .graphql_client.input_types import ArgumentInput, ModuleInstanceInput, ModuleInstanceResourcesInput
from .graphql_client.latest_modules import LatestModulesLatestModulesPageInfo
from .graphql_client.module_instance_details import ModuleInstanceDetailsModuleInstance
from .graphql_client.module_instance_full import (
    ModuleInstanceFullModuleInstance,
)
from .graphql_client.module_instances import (
    ModuleInstancesMeAccountModuleInstancesEdgesNode,
    ModuleInstancesMeAccountModuleInstancesPageInfo,
)
from .graphql_client.modules import ModulesModulesPageInfo
from .graphql_client.retry import RetryRetry
from .graphql_client.run import RunRun
from .typedef import build_typechecker, type_from_typedef

ArgId = UUID
ModuleInstanceId = UUID
Target = ModuleInstanceTarget


@dataclass
class ModuleInstanceHistory:
    path: str
    id: ModuleInstanceId
    status: ModuleInstanceStatus
    tags: list[str]


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
    edges = []


class TenguModuleRunner(Protocol[TCo]):
    async def __call__(
        self,
        *args: Any,
        target: Target,
        resources: ModuleInstanceResourcesInput | None = None,
        tags: list[str] | None = None,
        restore: bool | None = None,
    ) -> TCo:
        ...


def get_name_from_path(path: str):
    return path.split("#")[-1].replace("_tengu", "").replace("tengu_", "")


class BaseProvider:
    """
    A class representing a provider for the Tengu quantum chemistry workflow platform.
    """

    class Arg(Generic[T]):
        def __init__(
            self,
            provider: "BaseProvider | None",
            id: UUID | None = None,
            value: T | None = None,
            typeinfo: dict[str, Any] | None = None,
        ):
            self.provider = provider
            self.id = id
            self.value = value
            self.typeinfo = typeinfo

        def __repr__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __str__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __eq__(self, other: "Provider.Arg[Any]"):
            return self.id == other.id

        async def info(self) -> ArgumentArgument:
            if self.id is None:
                raise Exception("No ID provided")
            if self.provider is None:
                raise Exception("No provider provided")

            return await self.provider.argument(self.id)

        async def download(
            self,
            filename: str | None = None,
            filepath: Path | None = None,
        ):
            if self.id is None:
                raise Exception("No ID provided")
            if self.provider is None:
                raise Exception("No provider provided")
            if self.typeinfo is None:
                await self.get()

            if self.typeinfo:
                if self.typeinfo["k"] == "object" or (
                    self.typeinfo["k"] == "optional" and self.typeinfo["t"]["k"] == "object"
                ):
                    await self.provider.download_object(self.id, filename, filepath)
                else:
                    raise Exception("Cannot download non-object argument")
            else:
                raise Exception("Cannot download argument without typeinfo")

        async def get(self) -> T:
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
                        remote_arg = await self.provider.argument(self.id)
                        self.typeinfo = remote_arg.typeinfo
                        if remote_arg.rejected_at:
                            if remote_arg.source:
                                # get the failure reason by checking the source module instance
                                module_instance = await self.provider.module_instance(remote_arg.source)
                                if (
                                    module_instance.stderr
                                    and module_instance.stderr.edges
                                    and len(module_instance.stderr.edges) > 0
                                ):
                                    raise Exception(module_instance.stderr.edges[-1].node.content)
                            raise Exception("Argument was rejected")
                        else:
                            self.value = remote_arg.value
                            if self.value is None:
                                time.sleep(1)
                    except GraphQLClientGraphQLMultiError as e:
                        if e.errors[0].message == "not found":
                            time.sleep(1)
                        else:
                            print(e.errors)
                            raise e

            if self.typeinfo and self.provider and self.id:
                if self.typeinfo["k"] == "object" or (
                    self.typeinfo["k"] == "optional" and self.typeinfo["t"]["k"] == "object"
                ):
                    return await self.provider.object(self.id)
            return self.value

    def __init__(
        self, client: Client, workspace: str | Path | None = None, batch_tags: list[str] | None = None
    ):
        """
        Initialize the TenguProvider a graphql client.
        """

        self.history = None
        self.client = client
        self.module_paths: dict[str, str] = {}
        if workspace:
            self.workspace = Path(workspace)
            self.restore(workspace)
        else:
            self.workspace = None

        if not self.history:
            self.history = History(tags=batch_tags or [], instances=[])
        self.batch_tags = batch_tags or []

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

    async def nuke(self, remote: bool = False):
        """
        Delete the workspace, and optionally the data stored for it on the server.
        """
        # first untrack the runs remotely if necessary
        if remote:
            if self.history:
                for instance in self.history.instances:
                    await self.delete_module_instance(instance.id)
        if self.workspace:
            for f in self.workspace.glob("*"):
                if f.is_dir():
                    for ff in f.glob("*"):
                        ff.unlink()
                    f.rmdir()
                else:
                    f.unlink()
            self.workspace.rmdir()

    def restore(self, workspace: str | Path):
        """
        Restore the workspace.
        """
        self.workspace = Path(workspace)
        # read the workspace history file
        # if it exists, load the history
        workspace_history = self.workspace / "history.json"
        if workspace_history.exists():
            self.history = self._load_history(workspace_history)

        if (self.workspace / "tengu.lock").exists():
            self.load_module_paths(self.workspace / "tengu.lock")

    def save(self, history_file: str | Path | None = None):
        """
        Save the workspace.
        """
        if self.workspace is None:
            raise Exception("No workspace provided")
        if history_file is None:
            history_file = self.workspace / "history.json"
        self.save_module_paths(self.module_paths, self.workspace / "tengu.lock")
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
                str(instance.id): (instance.status, get_name_from_path(instance.path), 1)
                for instance in instances
            }

        if group_by == "path":
            c = Counter([(instance.status, get_name_from_path(instance.path)) for instance in instances])
            return {f"{name} ({status})": (status, name, count) for (status, name), count in c.items()}

        if group_by == "tag":
            c = Counter([(instance.status, instance.tags) for instance in instances])
            return {f"{tag} ({status})": (status, "", count) for (status, tag), count in c.items()}

        raise Exception("Invalid group_by")

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
            if len(result.edges) > 0:
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

    async def object(self, id: ArgId):
        """
        Retrieve an object from the database.

        :param id: The ID of the object.
        :return: The object.
        """
        return await self.client.object(id)

    async def download_object(self, id: ArgId, filename: str | None = None, filepath: Path | None = None):
        """
        Retrieve an object from the store: a wrapper for object with simpler behavior.

        :param id: The ID of the object.
        :param filepath: Where to download the object.
        :param filename: Download to the workspace with this name under "objects".
        """
        obj = await self.object(id)

        if filepath is None:
            if filename is None:
                filename = str(id)
            if filename and self.workspace:
                if not (self.workspace / "objects").exists():
                    (self.workspace / "objects").mkdir()
                filepath = self.workspace / "objects" / filename

                # we only error on the implicit path, if the user provides filepath we will overwrite
                if filepath.exists():
                    raise FileExistsError(f"File {filename} already exists in workspace")

        if filepath:
            if "url" in obj:
                with httpx.stream(method="get", url=obj["url"]) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
            else:
                with open(filepath, "w") as f:
                    json.dump(obj, f)
        else:
            return obj

    def load_module_paths(self, filepath: Path) -> dict[str, str]:
        """
        Load all of the module versions from a file.

        :param filename: Json module version file
        """
        modules = None
        if filepath.exists() and filepath.stat().st_size > 0:
            with open(filepath, "r") as f:
                modules = json.load(f)
            self.module_paths = modules
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

    async def run(
        self,
        path: str,
        args: list[Arg[Any] | Argument | ArgId | Path | IOBase | Any],
        target: Target | None = None,
        resources: ModuleInstanceResourcesInput | None = None,
        tags: list[str] | None = None,
        out_tags: list[list[str] | None] | None = None,
        restore: bool | None = None,
    ) -> RunRun | ModuleInstanceFullModuleInstance:
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
        tags = tags + self.batch_tags if tags else self.batch_tags

        if restore:
            res: list[ModuleInstanceFullModuleInstance] = []
            async for page in await self.module_instances(tags=tags, path=path):
                for edge in page.edges:
                    instance = edge.node
                    res.append(instance)
                    if len(res) > 1:
                        print("Warning: multiple module instances found with the same tags and path")
                        break
            if len(res) == 1:
                return res[0]

        storage_requirements = {"storage": 0}

        # TODO: less insane version of this
        def gen_arg_dict(input: Provider.Arg[Any] | ArgId | UUID | Path | IOBase | Any) -> ArgumentInput:
            arg = ArgumentInput()
            if isinstance(input, Provider.Arg):
                if input.id is None:
                    arg.value = input.value
                else:
                    arg.id = input.id
            elif isinstance(input, ArgId):
                arg.id = input
            elif isinstance(input, Path):
                storage_requirements["storage"] += input.stat().st_size
                if input.name.endswith(".json"):
                    with open(input, "r") as f:
                        arg = ArgumentInput(value=json.load(f))
                else:
                    arg = ArgumentInput(value=base64.b64encode(input.read_bytes()).decode("utf-8"))
            elif isinstance(input, IOBase):
                data = input.read()
                # The only other case is bytes-like, i.e. isinstance(data, (bytes, bytearray))
                if isinstance(data, str):
                    data = data.encode("utf-8")
                arg = ArgumentInput(value=base64.b64encode(data).decode("utf-8"))
            else:
                arg = ArgumentInput(value=input)
            return arg

        arg_dicts = [gen_arg_dict(input) for input in args]

        if resources is None and storage_requirements["storage"] > 0:
            resources = ModuleInstanceResourcesInput(
                storage=int(math.ceil(storage_requirements["storage"] / 1024)), storage_units=MemUnits.MB
            )

        runres = await self.client.run(
            ModuleInstanceInput(
                path=path,
                args=arg_dicts,
                target=target,
                resources=resources,
                tags=tags,
                out_tags=out_tags,
            )
        )
        if not self.history:
            self.history = History(tags=tags, instances=[])
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
        first: Union[Optional[int], UnsetType] = UNSET,
        after: Union[Optional[str], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        path: Union[Optional[str], UnsetType] = UNSET,
        name: Union[Optional[str], UnsetType] = UNSET,
        status: Union[Optional[ModuleInstanceStatus], UnsetType] = UNSET,
        tags: Union[Optional[List[str]], UnsetType] = UNSET,
        ids: Union[Optional[List[ModuleInstanceId]], UnsetType] = UNSET,
    ) -> AsyncIterable[
        Page[ModuleInstanceFullModuleInstance, ModuleInstancesMeAccountModuleInstancesPageInfo]
    ]:
        """
        Retrieve a list of module instancees filtered by the given parameters.

        :param first: Retrieve the first N module instances.
        :param after: Retrieve module instances after a certain cursor.
        :param last: Retrieve the last N module instances.
        :param before: Retrieve module instances before a certain cursor.
        :param path: Retrieve module instancees with for the given module path.
        :param name: Retrieve module instancees with for the given module name.
        :param status: Retrieve module instancees with the specified status ("CREATED", "ADMITTED", "QUEUED", "DISPATCHED", "COMPLETED", "FAILED").
        :param tags: Retrieve module instancees with the given list of tags.
        :return: A list of filtered module instancee.
        """

        async def return_paged(
            after: Union[Optional[str], UnsetType] = UNSET,
            before: Union[Optional[str], UnsetType] = UNSET,
            first: Union[Optional[int], UnsetType] = UNSET,
            last: Union[Optional[int], UnsetType] = UNSET,
            **kwargs: Any,
        ) -> Page[
            ModuleInstancesMeAccountModuleInstancesEdgesNode, ModuleInstancesMeAccountModuleInstancesPageInfo
        ]:
            res = await self.client.module_instances(
                first=first, after=after, last=last, before=before, **kwargs
            )
            # FIXME: this passes in mypy but not in pyright
            return res.account.module_instances  # type: ignore

        return self._query_with_pagination(
            return_paged,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {"path": path, "name": name, "status": status, "tags": tags, "ids": ids},
        )

    async def modules(
        self,
        first: Union[Optional[int], UnsetType] = UNSET,
        after: Union[Optional[str], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        path: Union[Optional[str], UnsetType] = UNSET,
        tags: Union[Optional[List[str]], UnsetType] = UNSET,
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

        return self._query_with_pagination(
            self.client.modules,  # type: ignore
            PageVars(after=after, before=before, first=first, last=last),
            {
                "path": path,
                "tags": tags,
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

    async def get_latest_module_paths(self, names: list[str] | None = None) -> dict[str, str]:
        """
        Get the latest module paths for a list of modules.

        :param names: The names of the modules.
        """
        ret = {}
        module_pages = await self.latest_modules(names=names)
        async for module_page in module_pages:
            for module in module_page.edges:
                path = module.node.path
                name = get_name_from_path(module.node.path)
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
            async for m in ms:
                yield m

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
        return await self.client.tag(
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
        return await self.client.untag(
            tags=tags,
            moduleId=module_id,
            moduleInstanceId=module_instance_id,
            argumentId=argument_id,
        )

    async def get_module_functions(
        self, names: list[str] | None = None, lockfile: Path | None = None
    ) -> dict[str, TenguModuleRunner[Any]]:
        """
        Get a dictionary of module functions.

        :param names: The names of the modules.
        :param lockfile: The lockfile to load the modules from.
        :return: A dictionary of module functions.
        """

        ret: dict[str, TenguModuleRunner[Any]] = {}
        if lockfile is not None:
            module_paths = self.load_module_paths(lockfile)
            module_pages = self.get_modules_for_paths(list(module_paths.values()))
        else:
            if self.workspace:
                if self.module_paths.items():
                    # we have already loaded a lock via the workspace
                    module_pages = self.get_modules_for_paths(list(self.module_paths.values()))
                else:
                    # lets load the latest paths and lock them
                    paths = await self.get_latest_module_paths(names)
                    module_pages = self.get_modules_for_paths(list(paths.values()))
                    self.module_paths = paths
                    self.save_module_paths(self.module_paths, self.workspace / "tengu.lock")
            else:
                # no workspace, so up the user to lock it
                module_pages = await self.latest_modules(names=names)

        async for page in module_pages:
            for edge in page.edges:
                module = edge.node.__deepcopy__()
                path = module.path
                in_types = tuple([type_from_typedef(i) for i in module.ins])
                out_types = tuple([type_from_typedef(i) for i in module.outs])

                typechecker = build_typechecker(*in_types)

                default_target = None
                if module.targets:
                    default_target = module.targets[0]

                default_resources = None
                if module.resource_bounds:
                    default_resources = ModuleInstanceResourcesInput(
                        storage=module.resource_bounds.storage_min,
                        storage_units=MemUnits.MB,
                        gpus=module.resource_bounds.gpu_hint,
                    )

                name = get_name_from_path(module.path)

                def closure(
                    name: str,
                    path: str,
                    typechecker: Any,
                    default_target: Target | None,
                    default_resources: ModuleInstanceResourcesInput | None,
                ):
                    async def runner(
                        *args: Any,
                        target: Target | None = default_target,
                        resources: ModuleInstanceResourcesInput | None = default_resources,
                        tags: list[str] | None = None,
                        restore: bool | None = None,
                    ):
                        typechecker(*args)
                        run = await self.run(
                            path, list(args), target, resources, tags, out_tags=None, restore=restore
                        )
                        outs: list[Any] = []
                        for out in run.outs:
                            outs.append(Provider.Arg(self, out.id))
                        return outs

                    runner.__name__ = name

                    # convert ins_usage array to argument docs
                    ins_docs = ""
                    if module.ins_usage:
                        for ins in module.ins_usage:
                            ins_docs += f"\n:param {ins}"

                    # convert outs_usage array to return docs
                    outs_docs = ""
                    if module.outs_usage:
                        for outs in module.outs_usage:
                            outs_docs += f"\n:return {outs}"

                    if module.description:
                        runner.__doc__ = (
                            module.description
                            + "\n\nModule version: "
                            + path
                            + "\n\nQDX Type Description:\n\n    "
                            + module.typedesc.replace(",", ",\n\n    ")
                            .replace("; ", ";\n\n    ")
                            .replace("}", "\n\n    }")
                            .replace("{", "{\n\n    ")
                            .replace("-> ", "\n\n->\n\n    ")
                            + (module.usage if module.usage else "")
                            + "\n\n\n"
                            + (ins_docs)
                            + (outs_docs)
                        )
                    else:
                        runner.__doc__ = name + " @" + path

                    runner.__annotations__["return"] = [t.to_python_type() for t in out_types]
                    runner.__annotations__["args"] = [t.to_python_type() for t in in_types]

                    return runner

                runner = closure(name, path, typechecker, default_target, default_resources)
                self.__setattr__(name, runner)
                ret[name] = runner
        return ret

    async def retry(
        self,
        id: ModuleInstanceId,
        target: Target,
        resources: ModuleInstanceResourcesInput | None = None,
    ) -> RetryRetry:
        """
        Retry a module instance.

        :param id: The ID of the module instance to be retried.
        :return: The ID of the new module instance.
        """
        return await self.client.retry(instance=id, resources=resources, target=target)

    async def upload(
        self,
        file: Path | str,
        typeinfo: dict[str, Any],
    ):
        """
        Upload an Object with typeinfo and store as an Argument.

        :param file: The file to be uploaded.
        :param typeinfo: The typeinfo of the file.
        """
        with open(file, "rb") as f:
            return await self.client.upload_arg(
                typeinfo=typeinfo,
                file=Upload(filename=f.name, content=f, content_type="application/octet-stream"),
            )

    async def module_instance(self, id: ModuleInstanceId) -> ModuleInstanceDetailsModuleInstance:
        """
        Retrieve a module instance by its ID.

        :param id: The ID of the module instance to be retrieved.
        :return: The retrieved module instance.
        :raise Exception: If the module instance is not found.
        """
        return await self.client.module_instance_details(id)

    async def logs(
        self,
        id: ModuleInstanceId,
        kind: Literal["stdout", "stderr"],
        after: Optional[str] = None,
        before: Optional[str] = None,
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
            return_paged, PageVars(after=after, before=before), {}  # type: ignore
        ):
            for edge in page.edges:
                yield edge.node.content
                i += 1
                if pages is not None and i > pages:
                    return

    async def delete_module_instance(self, id: ModuleInstanceId):
        """
        Delete a module instance with a given ID.

        :param id: The ID of the module instance to be deleted.
        :return: The ID of the deleted module instance.
        :raise RuntimeError: If the operation fails.
        """
        return await self.client.delete_module_instance(id)

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
        :raise Exception: If the module instance fails or polling times out.
        """
        n_try = 0

        curr_poll_rate = 0.5
        while n_try < n_retries:
            time.sleep(curr_poll_rate)
            if curr_poll_rate == poll_rate:
                n_try += 1
            curr_poll_rate = min(curr_poll_rate * 2, poll_rate)
            module_instance = await self.client.module_instance_minimal(id=id)
            if module_instance and module_instance.status in ["COMPLETED", "FAILED"]:
                return module_instance

        raise Exception("Module polling timed out")


class Provider(BaseProvider):
    def __init__(
        self,
        access_token: str | None = None,
        url: str | None = None,
        workspace: str | Path | None = None,
        batch_tags: list[str] | None = None,
    ):
        """
        Initialize the TenguProvider with a graphql client.

        :param access_token: The access token to use.
        :param url: The url to use.
        :param workspace: The workspace directory to use.
        :param batch_tags: The tags that will be placed on all runs by default.
        """
        if access_token or url is None:
            # try to check the environment variables
            import os

            if access_token is None:
                access_token = os.environ.get("TENGU_TOKEN")
                if access_token is None:
                    raise Exception("No access token provided")

            if url is None:
                url = os.environ.get("TENGU_URL")
                if url is None:
                    raise Exception("No url provided")
            client = Client(url=url, headers={"authorization": f"bearer {access_token}"})
            super().__init__(client, workspace=workspace, batch_tags=batch_tags)
        else:
            client = Client(url=url, headers={"authorization": f"bearer {access_token}"})
            super().__init__(client, workspace=workspace, batch_tags=batch_tags)


async def build_provider_with_functions(
    access_token: str | None = None,
    url: str | None = None,
    workspace: str | Path | None = None,
    batch_tags: list[str] | None = None,
    module_names: list[str] | None = None,
) -> Provider:
    """
    Build a TenguProvider with the given access token and url.

    :param access_token: The access token to use.
    :param url: The url to use.
    :param workspace: The workspace directory to use.
    :param batch_tags: The tags that will be placed on all runs by default.
    :return: The built TenguProvider.
    """
    provider = Provider(access_token, url, workspace, batch_tags)

    await provider.get_module_functions(names=module_names)
    return provider
