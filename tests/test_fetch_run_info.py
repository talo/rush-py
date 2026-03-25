from rush.client import (
    RunID,
    RushRunInfo,
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
                {"target": "Bullet", "sus": 5},
                {"target": "Gadi", "sus": 12},
                {"target": "Setonix", "sus": 8},
                {"target": "Gadi", "sus": 3},
                {"target": "Setonix", "sus": None},
            ]
        },
        {
            "nodes": [
                {"target": "Bullet"},
                {"target": "Gadi"},
                {"target": "Setonix"},
            ]
        },
    ) == {"Gadi": 15, "Setonix": 8}


def test_run_sus_returns_zero_for_supported_targets_with_no_usage():
    assert _run_sus(
        None,
        {
            "nodes": [
                {"target": "Gadi"},
                {"target": "Setonix"},
                {"target": "Bullet"},
            ]
        },
    ) == {"Gadi": 0, "Setonix": 0}


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
                            {"target": "Gadi"},
                        ]
                    },
                    "resource_utilizations": {
                        "nodes": [
                            {"target": "Gadi", "walltime": 7, "sus": 2.5},
                            {"target": "Gadi", "walltime": 13, "sus": 1.5},
                            {"target": "Bullet", "walltime": 5, "sus": 99},
                        ]
                    },
                }
            }

    monkeypatch.setattr("rush.client._get_client", lambda: FakeClient())

    info = fetch_run_info("run-id")

    assert info is not None
    assert info.id == RunID("run-id")
    assert info.walltime == 25
    assert info.sus == {"Gadi": 4.0}
    assert "walltime:    25" in str(info)
    assert "Gadi SUs:  4.0" in str(info)


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
                            {"target": "Gadi"},
                        ]
                    },
                    "resource_utilizations": {"nodes": []},
                }
            }

    monkeypatch.setattr("rush.client._get_client", lambda: FakeClient())

    info = fetch_run_info("run-id")

    assert info is not None
    assert info.walltime == 0
    assert info.sus == {"Gadi": 0}
    assert "walltime:    0 (incomplete)" in str(info)
    assert "Gadi SUs:  0 (incomplete)" in str(info)


def test_run_info_marks_zero_resource_totals_incomplete_for_non_final_runs():
    info = RushRunInfo(
        id=RunID("run-id"),
        created_at="2026-03-24T10:00:00.000000",
        updated_at="2026-03-24T10:20:00.000000",
        status="running",
        walltime=0,
        sus={"Gadi": 0},
    )

    formatted = str(info)

    assert "walltime:    0 (incomplete)" in formatted
    assert "Gadi SUs:  0 (incomplete)" in formatted
