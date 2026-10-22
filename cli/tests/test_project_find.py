import pytest

from conftest import log_in_as

PID = "7f3c1e2a-0b1d-4c55-9d3e-123456789abc"
PROJECT = {"project_id": PID, "name": "Vision", "description": "Image classifier"}
BASE = f"/mlflow/projects/{PID}"


@pytest.fixture(autouse=True)
def signed_in(server):
    log_in_as()
    server.on("GET", "/mlflow/projects", 200, [PROJECT])


def test_find_shows_project_id_description_and_version_state(run, server, capsys):
    server.on("GET", f"{BASE}/versions", 200, [
        {"version": "B", "is_current": False, "is_head": True},
        {"version": "A", "is_current": True, "is_head": False},
    ])
    assert run("project", "find", "vision") == 0
    output = capsys.readouterr().out
    assert f"Project ID: {PID}" in output
    assert "Description: Image classifier" in output
    assert "Versions: 2" in output
    assert "Current version: A" in output
    assert "Newest version: B" in output


def test_find_without_versions_explains_linking(run, server, capsys):
    server.on("GET", f"{BASE}/versions", 200, [])
    assert run("project", "find", "Vision") == 0
    output = capsys.readouterr().out
    assert "Versions: none uploaded yet." in output
    assert "prayog project link Vision FOLDER" in output


def test_find_unknown_project_explains_create(run, server, capsys):
    server.on("GET", "/mlflow/projects", 200, [])
    assert run("project", "find", "Nope") == 1
    assert "prayog project create Nope" in capsys.readouterr().err
