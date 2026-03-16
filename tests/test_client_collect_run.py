from rush.client import RunError, collect_run


def test_collect_run_restored(monkeypatch, capsys):
    monkeypatch.setattr(
        "rush.client._poll_run",
        lambda run_id, max_wait_time: ("done", False),
    )
    monkeypatch.setattr(
        "rush.client._fetch_results",
        lambda run_id: {
            "status": "done",
            "result": [{"path": "output.json"}],
            "trace": "",
        },
    )

    result = collect_run("run-id")

    assert result == {"path": "output.json"}
    assert "Restored already-completed run" in capsys.readouterr().err


def test_collect_run_no_mi_error(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "rush.client._poll_run",
        lambda run_id, max_wait_time: ("error", False),
    )
    monkeypatch.setattr(
        "rush.client._fetch_results",
        lambda run_id: {
            "status": "error",
            "result": (
                "Module instance creation failed: module `exess_rex` is not "
                "available for this account tier"
            ),
            "trace": (
                'stdout: Some("starting rex evaluation")\\n'
                'stderr: Some("module `exess_rex` is not available for this account tier")'
            ),
        },
    )

    result = collect_run("run-id")

    stderr = capsys.readouterr().err
    assert isinstance(result, RunError)
    assert "module `exess_rex` is not available" in result.message
    assert "starting rex evaluation" in stderr
    assert "Restored already-completed run" not in stderr
