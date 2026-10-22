import pytest

from conftest import log_in_as

PID = "7f3c1e2a-0b1d-4c55-9d3e-123456789abc"
PROJECT = {"project_id": PID, "name": "Vision"}
BASE = f"/mlflow/projects/{PID}"
REBUILT = {
    "run_id": 2, "label": "A run 2", "version": "A", "run_number": 2,
    "job_id": 42, "status": "QUEUED", "job_state": "IN_PROCESS",
    "queue_reason": None, "gpus": 1, "walltime": "02:00",
    "hyperparameters": {}, "dataset": [], "checkpoints": {}, "metrics": {},
    "logs_artifact_id": None, "snapshot_id": None, "error_summary": None,
    "created_at": "2026-10-22T10:00:00", "started_at": None,
    "finished_at": None, "created": True, "rebuild_of": "A/1",
}


@pytest.fixture(autouse=True)
def signed_in(server):
    log_in_as()
    server.on("GET", "/mlflow/projects", 200, [PROJECT])


def test_rebuild_command_uses_existing_rebuild_submission_flow(run, server, capsys):
    server.on("POST", f"{BASE}/runs", 201, REBUILT)
    assert run("rebuild", "Vision", "A/1", "--walltime", "02:00", "--gpus", "1") == 0
    body = server.body()
    assert body["rebuild_of"] == "A/1"
    assert body["gpus"] == 1 and body["walltime"] == "02:00"
    assert "entrypoint" not in body and "hyperparameters" not in body and "version" not in body
    output = capsys.readouterr().out
    assert "a rebuild of A/1" in output
    assert "prayog project compare Vision A/2" in output


def test_rebuild_command_rejects_invalid_gpu_count_before_submission(run, server, capsys):
    assert run("rebuild", "Vision", "A/1", "--walltime", "02:00", "--gpus", "0") == 2
    assert "--gpus must be at least 1" in capsys.readouterr().err
    assert not any(request.url.path.endswith("/runs") and request.method == "POST" for request in server.seen)


def test_rebuild_command_preserves_server_rejection_reason(run, server, capsys):
    server.on("POST", f"{BASE}/runs", 404, {
        "detail": "This run cannot be rebuilt: it was not found, or has no verified snapshot.",
        "code": "REBUILD_REJECTED",
    })
    assert run("rebuild", "Vision", "A/9", "--walltime", "02:00") == 1
    assert "cannot be rebuilt" in capsys.readouterr().err
