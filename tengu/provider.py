import base64
import json
import math
import time
from dataclasses import dataclass
from io import IOBase
from pathlib import Path
from typing import Any, AsyncIterable, Generic, Iterable, List, Literal, Optional, Protocol, TypeVar, Union
from uuid import UUID

import httpx
from tengu.graphql_client.exceptions import GraphQLClientGraphQLMultiError

from tengu.graphql_client.module_instance_details import ModuleInstanceDetailsModuleInstance

from .graphql_client.argument import Argument, ArgumentArgument
from .graphql_client.arguments import (
    ArgumentsMeAccountArguments,
    ArgumentsMeAccountArgumentsEdgesNode,
    ArgumentsMeAccountArgumentsPageInfo,
)
from .graphql_client.base_model import UNSET, UnsetType, Upload
from .graphql_client.client import Client
from .graphql_client.enums import MemUnits, ModuleInstanceStatus, ModuleInstanceTarget
from .graphql_client.fragments import ModuleFull, PageInfoFull
from .graphql_client.input_types import ArgumentInput, ModuleInstanceInput, ModuleInstanceResourcesInput
from .graphql_client.latest_modules import LatestModulesLatestModulesPageInfo
from .graphql_client.module_instance_full import ModuleInstanceFull, ModuleInstanceFullModuleInstance
from .graphql_client.module_instances import (
    ModuleInstancesMeAccountModuleInstancesEdgesNode,
    ModuleInstancesMeAccountModuleInstancesPageInfo,
)
from .graphql_client.modules import ModulesModulesPageInfo
from .graphql_client.retry import RetryRetry
from .graphql_client.run import RunRun
from .typedef import build_typechecker, type_from_typedef

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
        target: ModuleInstanceTarget,
        resources: ModuleInstanceResourcesInput | None = None,
        tags: list[str] | None = None,
    ) -> TCo:
        ...


ArgId = UUID
ModuleInstanceId = UUID


class BaseProvider:
    """
    A class representing a provider for the Tengu quantum chemistry workflow platform.
    """

    class Arg(Generic[T]):
        def __init__(self, provider: "BaseProvider | None", id: UUID | None = None, value: T | None = None):
            self.provider = provider
            self.id = id
            self.value = value

        def __repr__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __str__(self):
            return f"Arg(id={self.id}, value={self.value})"

        def __eq__(self, other: "Provider.Arg[Any]"):
            return self.id == other.id

        async def get(self) -> T:
            """
            Get the value of the argument.

            This will wait until the argument is ready.

            :return: The value of the argument.
            """
            if self.value is None:
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

            if self.typeinfo and self.provider:
                if self.typeinfo["k"] == "object" or (
                    self.typeinfo["k"] == "optional" and self.typeinfo["t"]["k"] == "object"
                ):
                    return await self.provider.object(self.id)
            return self.value

    def __init__(self, client: Client):
        """
        Initialize the TenguProvider a graphql client.
        """

        self.client = client

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

    async def download_object(self, id: ArgId, filepath: Path):
        """
        Retrieve an object from the store: a wrapper for object with simpler behavior.

        :param id: The ID of the object.
        :param filepath: Where to download the object.
        """
        obj = await self.object(id)

        if "url" in obj:
            with httpx.stream(method="get", url=obj["url"]) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
        else:
            with open(filepath, "w") as f:
                json.dump(obj, f)

    def load_module_paths(self, filename: Path) -> dict[str, str]:
        """
        Load all of the module versions from a file.

        :param filename: Json module version file
        """
        modules = None
        with open(filename, "r") as f:
            modules = json.load(f)
        return modules

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
        args: list[Arg[Any] | Argument | Path | IOBase | Any],
        target: ModuleInstanceTarget | None = None,
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

        if restore:
            res: list[ModuleInstanceFullModuleInstance] = []
            async for page in await self.module_instances(tags=tags, path=path):
                for edge in page.edges:
                    instance = edge.node
                    res.append(instance.module_instance)
                    if len(res) > 1:
                        break
            if len(res) == 1:
                return res[0]

        storage_requirements = {"storage": 0}

        # TODO: less insane version of this
        def gen_arg_dict(input: Provider.Arg[Any] | ArgId | UUID | Path | IOBase | Any) -> ArgumentInput:
            arg = ArgumentInput()
            if isinstance(input, Provider.Arg):
                arg.id = input.id
                arg.value = input.value
            elif isinstance(input, ArgId):
                arg.id = input
            elif isinstance(input, Path):
                storage_requirements["storage"] += input.stat().st_size
                if input.name.endswith(".json"):
                    arg = ArgumentInput(value=json.loads(input.read_text()))
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

        return await self.client.run(
            ModuleInstanceInput(
                path=path,
                args=arg_dicts,
                target=target,
                resources=resources,
                tags=tags,
                out_tags=out_tags,
            )
        )

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
        **kwargs: Any,
    ) -> AsyncIterable[Page[ModuleInstanceFull, ModuleInstancesMeAccountModuleInstancesPageInfo]]:
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
            {
                "path": path,
                "name": name,
                "status": status,
                "tags": tags,
            },
        )

    async def modules(
        self,
        first: Union[Optional[int], UnsetType] = UNSET,
        after: Union[Optional[str], UnsetType] = UNSET,
        last: Union[Optional[int], UnsetType] = UNSET,
        before: Union[Optional[str], UnsetType] = UNSET,
        path: Union[Optional[str], UnsetType] = UNSET,
        name: Union[Optional[str], UnsetType] = UNSET,
        tags: Union[Optional[List[str]], UnsetType] = UNSET,
        **kwargs: Any,
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
                "name": name,
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
        **kwargs: Any,
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
                name = module.node.path.split("#")[-1]
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
            module_pages = await self.latest_modules(names=names)

        async for page in module_pages:
            for edge in page.edges:
                module = edge.node.__deepcopy__()
                path = module.path + ""
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

                name = module.path.split("#")[-1].replace("_tengu", "").replace("tengu_", "")

                def closure(
                    name: str,
                    path: str,
                    typechecker: Any,
                    default_target: ModuleInstanceTarget | None,
                    default_resources: ModuleInstanceResourcesInput | None,
                ):
                    async def runner(
                        *args: Any,
                        target: ModuleInstanceTarget | None = default_target,
                        resources: ModuleInstanceResourcesInput | None = default_resources,
                        tags: list[str] | None = None,
                    ):
                        typechecker(*args)
                        run = await self.run(path, list(args), target, resources, tags, out_tags=None)
                        outs: list[Any] = []
                        if isinstance(run, ModuleInstanceFull):
                            for out in run.module_instance.outs:
                                outs.append(Provider.Arg(self, out.id, out.value))
                        if isinstance(run, RunRun):
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
                            + "\n\nQDX Type Description:\n\n    "
                            + module.typedesc.replace(",", ",\n\n    ")
                            .replace("; ", ";\n\n    ")
                            .replace("}", "\n\n    }")
                            .replace("{", "{\n\n    ")
                            .replace("-> ", "\n\n->\n\n    ")
                            + (module.usage if module.usage else "")
                            + "\n\n"
                            + (ins_docs)
                            + (outs_docs)
                        )
                    else:
                        runner.__doc__ = name

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
        target: ModuleInstanceTarget,
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
            first: Union[Optional[int], UnsetType] = UNSET,
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

            return res.stderr if kind == "stderr" else res.stdout

        async for page in self._query_with_pagination(
            return_paged, PageVars(after=after, before=before), {}  # type: ignore
        ):
            for edge in page.edges:
                yield edge.node.content

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
    def __init__(self, access_token: str, url: str = "https://tengu.qdx.ai"):
        client = Client(url=url, headers={"authorization": f"bearer {access_token}"})
        super().__init__(client)


class BaseTypedProvider(BaseProvider):
    def __init__(self, client: Client):
        super().__init__(client)
        self._setup = self.get_module_functions()


class TypedProvider(BaseTypedProvider):
    def __init__(self, access_token: str, url: str = "https://tengu.qdx.ai"):
        client = Client(url=url, headers={"authorization": f"bearer {access_token}"})
        super().__init__(client)
