"""Shared pytest configuration for Rush job-submitting tests."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

import rush.session

# Test files that do not submit Rush jobs. New test files default to
# `submits_rush_jobs`, which is safer than accidentally bypassing queue-aware
# skipping and longer timeouts.
NON_SUBMITTING_TEST_FILES = frozenset(
    {
        "tests/module_output_helpers/test_auto3d_output_helpers.py",
        "tests/module_output_helpers/test_boltz_output_helpers.py",
        "tests/module_output_helpers/test_exess_output_helpers.py",
        "tests/module_output_helpers/test_hyper_output_helpers.py",
        "tests/module_output_helpers/test_mmseqs2_output_helpers.py",
        "tests/module_output_helpers/test_nnxtb_output_helpers.py",
        "tests/module_output_helpers/test_pbsa_output_helpers.py",
        "tests/module_output_helpers/test_prepare_output_helpers.py",
        "tests/test_client_collect_run.py",
        "tests/test_convert_mmcif.py",
        "tests/test_convert_pdb.py",
        "tests/test_exess_namespaces.py",
        "tests/test_fetch_runs.py",
        "tests/test_merge.py",
        "tests/test_fetch_run_info.py",
    }
)

# Threshold: if any target has more than this many queued+admitted jobs, skip slow tests.
QUEUE_BUSY_THRESHOLD = 2
TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TESTS_DIR / "data"


@dataclass
class QueueTarget:
    target: str
    created_count: int
    admitted_count: int
    queued_count: int
    running_count: int


def queue_status() -> list[QueueTarget]:
    """Query the Rush GraphQL API for current queue status."""
    endpoint = os.environ.get(
        "RUSH_ENDPOINT",
        "https://tengu-server-prod-api-519406798674.asia-southeast1.run.app",
    )
    token = os.environ.get("RUSH_TOKEN")
    if not token:
        raise RuntimeError("RUSH_TOKEN not set")

    resp = requests.post(
        endpoint,
        json={
            "query": """
                query { queue_status { target created_count admitted_count queued_count running_count } }
            """
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return [QueueTarget(**entry) for entry in data["data"]["queue_status"]]


def _queues_are_busy() -> bool:
    """Return True if any target queue has more than QUEUE_BUSY_THRESHOLD queued+admitted jobs."""
    try:
        statuses = queue_status()
    except Exception:
        # If we can't reach the API, assume busy and skip slow tests gracefully.
        return True

    for s in statuses:
        if s.queued_count + s.admitted_count > QUEUE_BUSY_THRESHOLD:
            return True
    return False


def _submits_rush_jobs(item: pytest.Item) -> bool:
    """Return True when a test submits jobs to Rush and should be queue-aware."""
    relative_path = item.path.resolve().relative_to(TESTS_DIR.parent).as_posix()
    return relative_path not in NON_SUBMITTING_TEST_FILES


# Cache the queue check result for the session.
_queue_busy: bool | None = None


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the canonical tests/data directory."""
    return TEST_DATA_DIR


@pytest.fixture(autouse=True)
def rush_workspace(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Provide a per-test Rush workspace under temp storage or a user-supplied root."""
    configured_root = request.config.getoption("--rush-workspace-dir")
    if configured_root:
        safe_nodeid = re.sub(r"[^A-Za-z0-9._-]+", "-", request.node.nodeid).strip("-")
        workspace = (
            Path(configured_root).expanduser().resolve()
            / safe_nodeid
            / "rush-workspace"
        )
    else:
        workspace = tmp_path / "rush-workspace"

    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture(autouse=True)
def _configure_rush_workspace(rush_workspace: Path):
    """Isolate each test's workspace so saved outputs never leak into the repo."""

    previous_workspace = rush.session._get_config().workspace_dir
    rush.session.configure(workspace_dir=rush_workspace)
    try:
        yield
    finally:
        rush.session.configure(workspace_dir=previous_workspace)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply queue-aware markers and timeouts to Rush job-submitting tests."""
    rush_job_marker = pytest.mark.submits_rush_jobs
    rush_job_timeout = pytest.mark.timeout(600)

    for item in items:
        if _submits_rush_jobs(item):
            item.add_marker(rush_job_marker)
            if not any(m.name == "timeout" for m in item.iter_markers()):
                item.add_marker(rush_job_timeout)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip Rush job-submitting tests when queues are busy."""
    global _queue_busy

    if "submits_rush_jobs" not in item.keywords:
        return

    # Allow forcing Rush job-submitting tests to run regardless of queue status.
    if item.config.getoption("--force-run-slow", default=False):
        return

    if _queue_busy is None:
        _queue_busy = _queues_are_busy()
        if _queue_busy:
            print("\n⚠ Rush queues are busy — skipping job-submitting tests")

    if _queue_busy:
        pytest.skip(f"Rush queues are busy (queued+admitted > {QUEUE_BUSY_THRESHOLD})")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Rush-specific pytest options."""
    parser.addoption(
        "--force-run-slow",
        action="store_true",
        default=False,
        help="Run Rush job-submitting tests even when Rush queues are busy",
    )
    parser.addoption(
        "--rush-workspace-dir",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Write Rush test workspaces under PATH, using a separate subdirectory "
            "for each test."
        ),
    )
