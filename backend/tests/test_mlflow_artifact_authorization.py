"""Artifact API ownership and response-header security tests."""
from tests.gpu_case import GpuRunCase, requires_postgres


@requires_postgres
class MLFlowArtifactAuthorizationTests(GpuRunCase):
    def test_only_the_artifact_owner_can_list_or_download_artifact_bytes(self):
        owner = self.make_user("artifact-owner")
        owner_client = self.client_as(owner)
        project = self.make_project(owner_client)
        artifact = self.upload(owner_client, project["experiment_id"], "CODE", b"private source archive")

        downloaded = owner_client.get(f"/mlflow/artifacts/{artifact['id']}/content")
        self.assertEqual(downloaded.status_code, 200, downloaded.text)
        self.assertEqual(downloaded.content, b"private source archive")
        self.assertEqual(downloaded.headers["cache-control"], "private, no-store")
        self.assertEqual(downloaded.headers["x-content-type-options"], "nosniff")

        other = self.make_user("artifact-other")
        other_client = self.client_as(other)
        denied_download = other_client.get(f"/mlflow/artifacts/{artifact['id']}/content")
        denied_list = other_client.get(f"/mlflow/experiments/{project['experiment_id']}/artifacts")
        self.assertEqual(denied_download.status_code, 404)
        self.assertEqual(denied_list.status_code, 404)
