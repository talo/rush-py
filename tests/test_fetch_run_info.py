from rush.runs import (
    RunID,
    RunInfo,
    _run_sus,
    _total_run_walltime,
    fetch_run_info,
)


def test_total_run_walltime_sums_resource_utilizations():
    assert (
        _total_run_walltime(
            {
                "nodes": [
                    {"walltime": 12},
                    {"walltime": None},
                    {"walltime": 18},
                ]
            }
        )
        == 30
    )


def test_total_run_walltime_returns_zero_when_present_but_unused():
    assert _total_run_walltime({"nodes": [{"walltime": None}, {"walltime": None}]}) == 0


def test_run_sus_only_includes_gadi_and_setonix():
    assert _run_sus(
        {
            "nodes": [
                {"target": "bullet", "sus": 5},
                {"target": "gadi", "sus": 12},
                {"target": "setonix", "sus": 8},
                {"target": "gadi", "sus": 3},
                {"target": "setonix", "sus": None},
            ]
        },
        {
            "nodes": [
                {"target": "bullet"},
                {"target": "gadi"},
                {"target": "setonix"},
            ]
        },
    ) == {"gadi": 15, "setonix": 8}


def test_run_sus_returns_zero_for_supported_targets_with_no_usage():
    assert _run_sus(
        None,
        {
            "nodes": [
                {"target": "gadi"},
                {"target": "setonix"},
                {"target": "bullet"},
            ]
        },
    ) == {"gadi": 0, "setonix": 0}


def test_fetch_run_info_includes_total_walltime(monkeypatch):
    class FakeClient:
        def execute(self, query):
            return {
                "run": {
                    "created_at": "2026-03-24T10:00:00.000000",
                    "deleted_at": None,
                    "updated_at": "2026-03-24T10:20:00.000000",
                    "name": "Example run",
                    "description": "Test run",
                    "tags": ["test"],
                    "result": {"ok": True},
                    "status": "done",
                    "trace": {},
                    "stdout": "done",
                    "module_instances": {
                        "nodes": [
                            {"target": "gadi"},
                        ]
                    },
                    "resource_utilizations": {
                        "nodes": [
                            {"target": "gadi", "walltime": 7, "sus": 2.5},
                            {"target": "gadi", "walltime": 13, "sus": 1.5},
                            {"target": "bullet", "walltime": 5, "sus": 99},
                        ]
                    },
                }
            }

    monkeypatch.setattr("rush.runs._get_client", lambda: FakeClient())

    info = fetch_run_info("run-id")

    assert info is not None
    assert info.id == RunID("run-id")
    assert info.walltime == 25
    assert info.sus == {"gadi": 4.0}
    assert "walltime:    25" in str(info)
    assert "Gadi SUs:    4.0" in str(info)


def test_fetch_run_info_seeds_supported_targets_before_sus_exist(monkeypatch):
    class FakeClient:
        def execute(self, query):
            return {
                "run": {
                    "created_at": "2026-03-24T10:00:00.000000",
                    "deleted_at": None,
                    "updated_at": "2026-03-24T10:20:00.000000",
                    "name": "Example run",
                    "description": None,
                    "tags": None,
                    "result": None,
                    "status": "pending",
                    "trace": None,
                    "stdout": None,
                    "module_instances": {
                        "nodes": [
                            {"target": "gadi"},
                        ]
                    },
                    "resource_utilizations": {"nodes": []},
                }
            }

    monkeypatch.setattr("rush.runs._get_client", lambda: FakeClient())

    info = fetch_run_info("run-id")

    assert info is not None
    assert info.walltime == 0
    assert info.sus == {"gadi": 0}
    assert "walltime:    0 (incomplete)" in str(info)
    assert "Gadi SUs:    0 (incomplete)" in str(info)


def test_run_info_marks_zero_resource_totals_incomplete_for_non_final_runs():
    info = RunInfo(
        id=RunID("run-id"),
        created_at="2026-03-24T10:00:00.000000",
        updated_at="2026-03-24T10:20:00.000000",
        status="running",
        walltime=0,
        sus={"gadi": 0},
    )

    formatted = str(info)

    assert "walltime:    0 (incomplete)" in formatted
    assert "Gadi SUs:    0 (incomplete)" in formatted
