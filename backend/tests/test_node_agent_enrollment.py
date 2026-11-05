"""Node identity and enrolment lifecycle tests (FR-GPU-110, 136)."""
from datetime import timedelta

from fastapi.testclient import TestClient

import models as orm
from logic.errors import BadRequestError
from logic.scheduler import nodes as node_ops
from logic.scheduler.domain import NodeState
from main import app
from scheduler_pg import SchedulerPostgresCase, requires_postgres


@requires_postgres
class NodeAgentEnrollmentTests(SchedulerPostgresCase):
    def _registration(self, label="agent"):
        return node_ops.register_node(
            self.db, f"{label}-{self.suffix}", self.lab.id, "shared", False, 2, now=self.now
        )

    def test_enrolment_token_is_one_time_and_only_its_hash_is_stored(self):
        registration = self._registration()
        node_id = registration.node.id

        self.assertNotEqual(registration.node.enrolment_token_hash, registration.enrolment_token)
        self.assertIsNone(registration.node.credential_hash)

        node, credential = node_ops.enrol(self.db, registration.enrolment_token, now=self.now)
        self.assertEqual(node.id, node_id)
        self.assertIsNone(node.enrolment_token_hash)
        self.assertIsNone(node.enrolment_expires_at)
        self.assertNotEqual(node.credential_hash, credential)
        self.assertIs(node_ops.authenticate(self.db, credential), node)

        with self.assertRaises(BadRequestError):
            node_ops.enrol(self.db, registration.enrolment_token, now=self.now)
        self.assertIsNone(node_ops.authenticate(self.db, registration.enrolment_token))

    def test_expired_and_unknown_enrolment_tokens_are_rejected(self):
        registration = self._registration("expired")
        with self.assertRaises(BadRequestError):
            node_ops.enrol(
                self.db, registration.enrolment_token,
                now=self.now + timedelta(hours=24, seconds=1),
            )
        with self.assertRaises(BadRequestError):
            node_ops.enrol(self.db, "not-a-real-token", now=self.now)

    def test_heartbeat_requires_a_valid_credential_and_enables_a_matching_node(self):
        registration = self._registration("heartbeat")
        client = TestClient(app)

        refused = client.post("/agent/heartbeat", json={"unit_count": 2, "running_allocation_ids": []},
                              headers={"X-Node-Credential": "invalid"})
        self.assertEqual(refused.status_code, 401)

        enrolled = client.post("/agent/enrol", json={"enrolment_token": registration.enrolment_token})
        self.assertEqual(enrolled.status_code, 200, enrolled.text)
        credential = enrolled.json()["credential"]
        node_id = enrolled.json()["node_id"]

        mismatch = client.post("/agent/heartbeat", json={"unit_count": 1, "running_allocation_ids": []},
                               headers={"X-Node-Credential": credential})
        self.assertEqual(mismatch.status_code, 200, mismatch.text)
        self.assertEqual(mismatch.json()["state"], NodeState.PENDING_AGENT)

        healthy = client.post("/agent/heartbeat", json={"unit_count": 2, "running_allocation_ids": []},
                              headers={"X-Node-Credential": credential})
        self.assertEqual(healthy.status_code, 200, healthy.text)
        self.assertEqual(healthy.json()["state"], NodeState.AVAILABLE)
        self.db.expire_all()
        node = self.db.get(orm.Node, node_id)
        self.assertEqual(node.state, NodeState.AVAILABLE)
        self.assertEqual(node.reported_unit_count, 2)

    def test_revoked_credential_cannot_authenticate_or_call_agent_api(self):
        registration = self._registration("revoked")
        node, credential = node_ops.enrol(self.db, registration.enrolment_token, now=self.now)
        node.credential_revoked_at = self.now
        self.db.commit()

        self.assertIsNone(node_ops.authenticate(self.db, credential))
        response = TestClient(app).get("/agent/allocations", headers={"X-Node-Credential": credential})
        self.assertEqual(response.status_code, 401)
