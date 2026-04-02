import inspect
import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import version as pkg_version
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

if TYPE_CHECKING:
    from .runs import RunID, RunOpts


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
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
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


GRAPHQL_ENDPOINT = (
    _get_env("RUSH_ENDPOINT")
    or "https://tengu-server-prod-api-519406798674.asia-southeast1.run.app"
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


MODULE_OVERRIDES = _get_env("RUSH_MODULE_LOCK")
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
        "smol_similarity_sumo": "github:talo/tengu-smol-similarity/d7ec9ca759d9c85eeafcd1afeb50b06a1d4b02e2#smol_similarity_sumo",
    }
    if "staging" in GRAPHQL_ENDPOINT
    else {
        # prod
        "auto3d_rex": "github:talo/tengu-auto3d/88c2fdc505f206463a9c60519273563b1dddabc9#auto3d_rex",
        "boltz2_rex": "github:talo/tengu-boltz2/76df0b4b4fa42e88928a430a54a28620feef8ea8#boltz2_rex",
        "exess_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_rex",
        "exess_geo_opt_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_geo_opt_rex",
        "exess_qmmm_rex": "github:talo/tengu-exess/133781d71c493900a82121729c18994b4a184197#exess_qmmm_rex",
        "mmseqs2_rex": "github:talo/tengu-colabfold/0b6ca8b9dc97fc6380d334169a6faae51d85fac7#mmseqs2_rex",
        "nnxtb_rex": "github:talo/tengu-nnxtb/4e733660264d38faab5d23eadc41ca86fd6ff97a#nnxtb_rex",
        "pbsa_rex": "github:talo/pbsa-cuda/f8b1c357fddfebf7e0c51a84f8d4e70958440c00#pbsa_rex",
        "prepare_protein_rex": "github:talo/tengu-prepare-protein/64dc3a9f37384508498c087f4c919673616302cc#prepare_protein_rex",
        "smol_similarity_sumo": "github:talo/tengu-smol-similarity/d7ec9ca759d9c85eeafcd1afeb50b06a1d4b02e2#smol_similarity_sumo",
    }
) | MODULE_OVERRIDES


_SDK_SESSION_ID = str(uuid.uuid4())


def _infer_sdk_function() -> str | None:
    """Infer which SDK function called _submit_rex() by walking the stack."""
    try:
        for frame_info in inspect.stack():
            module_path = frame_info.filename
            # Look for files in the rush package (but not session.py itself)
            if "/rush/" in module_path and "session.py" not in module_path:
                module_name = Path(module_path).stem
                func_name = frame_info.function
                return f"{module_name}.{func_name}"
    except Exception:
        pass
    return None


def _get_sdk_tags() -> list[str]:
    """Generate SDK metadata tags for run submission."""
    tags = ["source=rushpy"]

    try:
        version = pkg_version("rush-py")
        tags.append(f"sdk_version={version}")
    except Exception:
        pass

    # Unique per-process ID
    tags.append(f"sdk_session_id={_SDK_SESSION_ID}")

    # Python version
    tags.append(f"sdk_python={platform.python_version()}")

    # Platform (OS/arch)
    tags.append(f"sdk_platform={platform.system().lower()}/{platform.machine()}")

    # Infer which SDK function submitted this run
    sdk_function = _infer_sdk_function()
    if sdk_function:
        tags.append(f"sdk_function={sdk_function}")

    return tags


@dataclass
class _SessionConfig:
    """
    Settings to configure rush-py. Can be set through the `configure` function.
    """

    #: The directory where the workspace resides. (Default: current working directory)
    #: The history JSON file will be written here and the
    #: run outputs will be downloaded here (nested under a project folder).
    workspace_dir: Path = Path.cwd()


_config: _SessionConfig | None = None


def _get_config() -> _SessionConfig:
    global _config

    if _config is None:
        _config = _SessionConfig()

    return _config


def configure(*, workspace_dir: Path | None = None) -> None:
    """
    Configure process-wide Rush session settings.

    Currently, only allows setting the workspace directory.
    """
    config = _get_config()
    if workspace_dir is not None:
        config.workspace_dir = workspace_dir


_client: Client | None = None


def _get_client() -> Client:
    global _client

    if _client is None:
        _client = Client(
            transport=RequestsHTTPTransport(
                url=GRAPHQL_ENDPOINT,
                headers={"Authorization": f"Bearer {_get_api_key()}"},
            )
        )

    return _client


def _submit_rex(rex: str, run_opts: "RunOpts | None" = None) -> "RunID":
    from .runs import RunID, RunOpts

    if run_opts is None:
        run_opts = RunOpts()

    # Auto-generate SDK metadata tags
    merged_tags = (run_opts.tags or []) + _get_sdk_tags()

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
            "project_id": _get_project_id(),
        },
    }
    mutation.variable_values["input"] |= {
        k: v for k, v in asdict(run_opts_with_tags).items() if v is not None
    }

    result = _get_client().execute(mutation)
    run_id = RunID(result["eval"]["id"])
    created_at = result["eval"]["created_at"].split(".")[0]
    print(f"Run submitted @ {created_at} with ID: {run_id}", file=sys.stderr)

    history_filepath = _get_config().workspace_dir / "history.json"
    history_filepath.parent.mkdir(parents=True, exist_ok=True)

    matching_modules = [
        module
        for module in MODULE_LOCK
        if f"{module}_s" in rex or f"try_{module}" in rex
    ]
    if len(matching_modules) != 1:
        print(
            "Error: could not uniquely match submitted module, not adding to history",
            file=sys.stderr,
        )
        return run_id

    if history_filepath.exists():
        with history_filepath.open() as f:
            history = json.load(f)
    else:
        history = {"instances": []}

    module = matching_modules[0]
    history["instances"].append(
        {
            "run_id": run_id,
            "run_created_at": created_at,
            "module_path": MODULE_LOCK[module],
        }
    )

    with history_filepath.open("w") as f:
        json.dump(history, f, indent=2)

    return run_id
