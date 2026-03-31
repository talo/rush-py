import pytest

from rush import RunBackendError, RunModuleError, collect_run


def test_collect_run_restored(monkeypatch, capsys):
    monkeypatch.setattr(
        "rush.runs._poll_run",
        lambda run_id, max_wait_time: ("done", False),
    )
    monkeypatch.setattr(
        "rush.runs._fetch_results",
        lambda run_id: {
            "status": "done",
            "result": [{"path": "output.json"}],
            "trace": "",
        },
    )

    result = collect_run("run-id")

    assert result == [{"path": "output.json"}]
    assert "Restored already-completed run" in capsys.readouterr().err


def test_collect_run_no_mi_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "rush.runs._poll_run",
        lambda run_id, max_wait_time: ("error", False),
    )
    monkeypatch.setattr(
        "rush.runs._fetch_results",
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

    with pytest.raises(RunBackendError) as exc_info:
        collect_run("run-id")

    stderr = capsys.readouterr().err
    assert "module `exess_rex` is not available" in exc_info.value.message
    assert "starting rex evaluation" in stderr
    assert "Restored already-completed run" not in stderr


def test_collect_run_prints_non_stream_trace_before_stdio(monkeypatch, capsys):
    monkeypatch.setattr(
        "rush.runs._poll_run",
        lambda run_id, max_wait_time: ("error", False),
    )
    monkeypatch.setattr(
        "rush.runs._fetch_results",
        lambda run_id: {
            "status": "error",
            "result": "rex evaluation failed",
            "trace": (
                'module_state: Some("rex_start_failed")\\n'
                'reason: Some("module not runnable on this account tier")\\n'
                'stdout: Some("starting rex evaluation")\\n'
                'stderr: Some("module `exess_rex` is not available for this account tier")'
            ),
        },
    )

    with pytest.raises(RunBackendError):
        collect_run("run-id")

    stderr = capsys.readouterr().err
    assert 'module_state: Some("rex_start_failed")' in stderr
    assert 'reason: Some("module not runnable on this account tier")' in stderr
    assert "stdout:" in stderr
    assert "stderr:" in stderr
    assert stderr.index("Trace:") < stderr.index("stdout:")
    assert stderr.index("stdout:") < stderr.index("stderr:")


def test_run_backend_error_str_includes_trace_and_stdio_sections():
    err = RunBackendError(
        "rex evaluation failed",
        (
            'module_state: Some("rex_start_failed")\\n'
            'stdout: Some("starting rex evaluation")\\n'
            'stderr: Some("module `exess_rex` is not available for this account tier")'
        ),
    )

    formatted = str(err)

    assert "Rush backend error: rex evaluation failed" in formatted
    assert "Contact QDX or submit a bug report" in formatted
    assert 'module_state: Some("rex_start_failed")' in formatted
    assert "stdout:" in formatted
    assert "stderr:" in formatted
    assert formatted.index("Trace:") < formatted.index("stdout:")


def test_collect_run_raises_run_module_error_for_inner_module_error(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "rush.runs._poll_run",
        lambda run_id, max_wait_time: ("done", True),
    )
    monkeypatch.setattr(
        "rush.runs._fetch_results",
        lambda run_id: {
            "status": "done",
            "result": {"Ok": {"Err": "Unsupported model configuration"}},
            "trace": "",
        },
    )

    with pytest.raises(RunModuleError) as exc_info:
        collect_run("run-id")

    stderr = capsys.readouterr().err
    assert exc_info.value.message == "Unsupported model configuration"
    assert "Run module error: Unsupported model configuration" in stderr
