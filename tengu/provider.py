import base64
import json
import time
from io import IOBase
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)
from uuid import UUID

import httpx
from tengu.graphql_client.module_instances import (
    ModuleInstancesMeAccountModuleInstances,
    ModuleInstancesMeAccountModuleInstancesPageInfo,
)

from tengu.graphql_client.run import RunRun

from .graphql_client.arguments import (
    ArgumentsMeAccountArguments,
    ArgumentsMeAccountArgumentsEdgesNode,
    ArgumentsMeAccountArgumentsPageInfo,
)
from .graphql_client.argument import Argument, ArgumentArgument
from .graphql_client.base_model import UNSET, UnsetType, Upload
from .graphql_client.client import Client
from .graphql_client.enums import ModuleInstanceStatus, ModuleInstanceTarget
from .graphql_client.fragments import ModuleFull, PageInfoFull
from .graphql_client.input_types import ArgumentInput, ModuleInstanceInput, ModuleInstanceResourcesInput
from .graphql_client.module_instance_full import ModuleInstanceFull
from .typedef import build_typechecker, type_from_typedef

T = TypeVar("T")
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

    def __init__(self, client: Client):
        """
        Initialize the TenguProvider a graphql client.
        """

        self.client = client

    async def _query_with_pagination(
        self,
        fn: Paged[T1, TPage],
        variables: dict[str, Any],
    ) -> AsyncIterable[Iterable[Edge[T1]]]:
        result = await fn(**variables)
        page_info_res = result.page_info
        yield result.edges or []

        while page_info_res.has_previous_page:
            new_vars = variables | {"before": page_info_res.end_cursor}
            result = await fn(**new_vars)
            page_info_res = result.page_info
            yield result.edges or []

    async def argument(self, id: ArgId) -> ArgumentArgument:
        """
        Retrieve an argument from the database.

        :param id: The ID of the argument.
        :return: The argument.
        """
        return await self.client.argument(id)

    async def arguments(
        self,
        first: int | None = None,
        after: str | None = None,
        last: int | None = None,
        before: str | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterable[ArgumentsMeAccountArguments]:
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
            return res.account.arguments

        return self._query_with_pagination(
            return_paged,
            {
                "first": first,
                "after": after,
                "last": last,
                "before": before,
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
    ) -> RunRun:
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
            res = []
            async for instance in await self.module_instances(tags=tags, path=path):
                res.append(instance)
                if len(res) > 1:
                    break
            if len(res) == 1:
                return res[0]

        # TODO: less insane version of this
        def gen_arg_dict(input: Provider.Arg[Any] | ArgId | UUID | Path | IOBase | Any) -> ArgumentInput:
            arg = ArgumentInput()
            if isinstance(input, Provider.Arg):
                arg.id = input.id
                arg.value = input.value
            elif isinstance(input, ArgId):
                arg.id = input
            elif isinstance(input, Path):
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
    ) -> AsyncIterable[ModuleInstanceFull]:
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
        ) -> Page[ModuleInstancesMeAccountModuleInstances, ModuleInstancesMeAccountModuleInstancesPageInfo]:
            res = await self.client.module_instances(
                first=first, after=after, last=last, before=before, **kwargs
            )
            return res.account.module_instances

        return self._query_with_pagination(
            return_paged,
            {
                "first": first,
                "after": after,
                "last": last,
                "before": before,
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
    ) -> AsyncIterable[list[Edge[ModuleFull]]]:
        """
        Get all modules.
        """

        return self._query_with_pagination(
            self.client.modules,
            {
                "first": first,
                "after": after,
                "last": last,
                "before": before,
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
    ) -> AsyncIterable[list[Edge[ModuleFull]]]:
        """
        Get latest modules.
        """

        return self._query_with_pagination(
            self.client.latest_modules,
            {
                "first": first,
                "after": after,
                "last": last,
                "before": before,
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
            for module in module_page:
                path = module.node.path
                name = module.node.path.split("#")[-1]
                if path:
                    ret[name] = path
        return ret

    async def get_modules_for_paths(self, paths: list[str]) -> AsyncIterable[list[Edge[ModuleFull]]]:
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
    ) -> dict[str, Callable[[list[Any]], list[Any]]]:
        ret = {}
        module_pages = []
        if lockfile is not None:
            module_paths = self.load_module_paths(lockfile)
            module_pages = self.get_modules_for_paths(list(module_paths.values()))
        else:
            module_pages = await self.latest_modules(names=names)

        async for page in module_pages:
            for edge in page:
                module = edge.node
                in_types = [type_from_typedef(i) for i in module.ins]
                out_types = [type_from_typedef(i) for i in module.outs]

                typechecker = build_typechecker(*in_types)

                def runner(
                    *args: Any,
                    target: ModuleInstanceTarget,
                    resources: ModuleInstanceResourcesInput | None = None,
                    tags: list[str] | None = None,
                ):
                    typechecker(*args)
                    return self.run(module.path, list(args), target, resources, tags, out_tags=None)

                runner.__name__ = module.path.split("#")[-1]

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
                        + (ins_docs)
                        + (outs_docs)
                    )
                else:
                    runner.__doc__ = module.path.split("#")[-1]
                runner.__annotations__["return"] = [t.to_python_type() for t in out_types]
                runner.__annotations__["args"] = [t.to_python_type() for t in in_types]
                name = module.path.split("#")[-1]
                self.__setattr__(name, runner)
                ret[name] = runner
        return ret

    async def retry(
        self,
        id: ModuleInstanceId,
        resources: ModuleInstanceResourcesInput | None = None,
        target: ModuleInstanceTarget | None = None,
    ) -> ModuleInstanceId:
        """
        Retry a module instance.

        :param id: The ID of the module instance to be retried.
        :return: The ID of the new module instance.
        """
        return await self.client.retry(id, resources, target)

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

    async def module_instance(self, id: ModuleInstanceId) -> Any:
        """
        Retrieve a module instance by its ID.

        :param id: The ID of the module instance to be retrieved.
        :return: The retrieved module instance.
        :raise Exception: If the module instance is not found.
        """
        return await self.client.module_instance_details(id)

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
    async def __init__(self, client: Client):
        super().__init__(client)
        await self.get_module_functions()


class TypedProvider(BaseTypedProvider):
    async def __init__(self, access_token: str, url: str = "https://tengu.qdx.ai"):
        client = Client(url=url, headers={"authorization": f"bearer {access_token}"})
        await super().__init__(client)
