import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")
os.environ.setdefault("POSTGRES_USER", "prayog")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "prayog_test")

from fastapi import HTTPException
from routers import node_agents
from schemas.node_agent import NodeEnrollmentIn, NodeHeartbeatIn


class NodeAgentEnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.admin = SimpleNamespace(id=7)

    def query_with_first(self, result):
        query = MagicMock()
        query.filter.return_value = query
        query.with_for_update.return_value = query
        query.first.return_value = result
        return query

    def test_enrollment_token_is_returned_once_but_only_hash_is_persisted(self):
        self.db.query.return_value = self.query_with_first(None)
        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            response = node_agents.create_enrollment_token(
                node_agents.EnrollmentTokenCreate(node_id="gpu-01", name="GPU 01"),
                self.db, self.admin,
            )
        record = self.db.add.call_args.args[0]
        self.assertEqual(response.node_id, "gpu-01")
        self.assertNotEqual(record.token_hash, response.enrollment_token)
        self.assertEqual(record.token_hash, node_agents._hash_secret(response.enrollment_token))
        self.assertEqual(record.created_by, 7)
        self.assertEqual(record.expires_at, datetime(2026, 1, 1) + timedelta(minutes=15))
        self.db.commit.assert_called_once()

    def test_admin_cannot_issue_another_token_for_an_active_node(self):
        self.db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
            node_id="gpu-01", revoked_at=None
        )
        with self.assertRaises(HTTPException) as raised:
            node_agents.create_enrollment_token(
                node_agents.EnrollmentTokenCreate(node_id="gpu-01", name="GPU 01"),
                self.db, self.admin,
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_called()

    def test_enrollment_rejects_wrong_node_consumed_and_expired_tokens(self):
        cases = [
            ("wrong-node", None, datetime(2026, 1, 1) + timedelta(minutes=5)),
            ("gpu-01", datetime(2025, 12, 31), datetime(2026, 1, 1) + timedelta(minutes=5)),
            ("gpu-01", None, datetime(2025, 12, 31)),
        ]
        for node_id, consumed_at, expires_at in cases:
            with self.subTest(node_id=node_id, consumed_at=consumed_at, expires_at=expires_at):
                self.db.reset_mock()
                enrollment = SimpleNamespace(node_id=node_id, consumed_at=consumed_at, expires_at=expires_at)
                self.db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = enrollment
                with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
                    with self.assertRaises(HTTPException) as raised:
                        node_agents.enroll_node_agent(
                            NodeEnrollmentIn(node_id="gpu-01", name="GPU 01", enrollment_token="t" * 40),
                            self.db,
                        )
                self.assertEqual(raised.exception.status_code, 401)
                event = self.db.add.call_args.args[0]
                self.assertEqual(event.outcome, "ENROLLMENT_REJECTED")
                self.assertNotIn("t" * 40, event.detail)
                self.db.commit.assert_called_once()

    def test_successful_enrollment_consumes_tokens_and_stores_only_credential_hash(self):
        enrollment = SimpleNamespace(
            node_id="gpu-01", consumed_at=None,
            expires_at=datetime(2026, 1, 1) + timedelta(minutes=5),
        )
        enrollment_query = self.query_with_first(enrollment)
        token_update_query = MagicMock()
        token_update_query.filter.return_value = token_update_query
        agent_query = self.query_with_first(None)
        self.db.query.side_effect = [enrollment_query, token_update_query, agent_query]
        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            response = node_agents.enroll_node_agent(
                NodeEnrollmentIn(node_id="gpu-01", name="GPU 01", enrollment_token="t" * 40),
                self.db,
            )
        self.assertEqual(response.node_id, "gpu-01")
        self.assertTrue(response.credential)
        self.assertEqual(enrollment.consumed_at, datetime(2026, 1, 1))
        token_update_query.update.assert_called_once()
        added = [call.args[0] for call in self.db.add.call_args_list]
        agent = next(item for item in added if isinstance(item, node_agents.NodeAgent))
        event = next(item for item in added if isinstance(item, node_agents.NodeAgentSecurityEvent))
        self.assertEqual(agent.credential_hash, node_agents._hash_secret(response.credential))
        self.assertNotEqual(agent.credential_hash, response.credential)
        self.assertEqual(event.outcome, "ENROLLED")
        self.assertNotIn(response.credential, repr(added))
        self.db.commit.assert_called_once()

    def test_active_agent_conflict_rolls_back_enrollment(self):
        enrollment = SimpleNamespace(
            node_id="gpu-01", consumed_at=None,
            expires_at=datetime(2026, 1, 1) + timedelta(minutes=5),
        )
        enrollment_query = self.query_with_first(enrollment)
        token_update_query = MagicMock()
        token_update_query.filter.return_value = token_update_query
        agent_query = self.query_with_first(SimpleNamespace(node_id="gpu-01", revoked_at=None))
        self.db.query.side_effect = [enrollment_query, token_update_query, agent_query]
        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            with self.assertRaises(HTTPException) as raised:
                node_agents.enroll_node_agent(
                    NodeEnrollmentIn(node_id="gpu-01", name="GPU 01", enrollment_token="t" * 40),
                    self.db,
                )
        self.assertEqual(raised.exception.status_code, 409)
        self.db.rollback.assert_called_once()
        self.db.commit.assert_not_called()



    def test_revoked_agent_can_reenroll_with_fresh_token_and_rotated_credential(self):
        enrollment = SimpleNamespace(
            node_id="gpu-01", consumed_at=None,
            expires_at=datetime(2026, 1, 1) + timedelta(minutes=5),
        )
        enrollment_query = self.query_with_first(enrollment)
        token_update_query = MagicMock()
        token_update_query.filter.return_value = token_update_query
        revoked_agent = SimpleNamespace(
            node_id="gpu-01",
            name="Old name",
            credential_hash="old-credential-hash",
            enrolled_at=datetime(2025, 1, 1),
            last_seen_at=datetime(2025, 1, 2),
            revoked_at=datetime(2025, 1, 3),
        )
        agent_query = self.query_with_first(revoked_agent)
        self.db.query.side_effect = [enrollment_query, token_update_query, agent_query]

        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            response = node_agents.enroll_node_agent(
                NodeEnrollmentIn(
                    node_id="gpu-01", name="GPU 01 renewed",
                    enrollment_token="n" * 40,
                ),
                self.db,
            )

        self.assertEqual(response.node_id, "gpu-01")
        self.assertEqual(revoked_agent.name, "GPU 01 renewed")
        self.assertEqual(
            revoked_agent.credential_hash,
            node_agents._hash_secret(response.credential),
        )
        self.assertNotEqual(revoked_agent.credential_hash, "old-credential-hash")
        self.assertEqual(revoked_agent.enrolled_at, datetime(2026, 1, 1))
        self.assertIsNone(revoked_agent.last_seen_at)
        self.assertIsNone(revoked_agent.revoked_at)
        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()

class NodeAgentHeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.credentials = SimpleNamespace(credentials="node-secret-credential")
        self.body = NodeHeartbeatIn(status="online", metadata={"gpu_count": 2})

    def set_agent(self, agent):
        self.db.query.return_value.filter.return_value.first.return_value = agent

    def test_heartbeat_accepts_matching_credential_and_updates_last_seen(self):
        agent = SimpleNamespace(node_id="gpu-01", revoked_at=None, last_seen_at=None)
        self.set_agent(agent)
        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            response = node_agents.report_heartbeat("gpu-01", self.body, self.credentials, self.db)
        self.assertTrue(response.accepted)
        self.assertEqual(response.node_id, "gpu-01")
        self.assertEqual(agent.last_seen_at, datetime(2026, 1, 1))
        event = self.db.add.call_args.args[0]
        self.assertEqual(event.outcome, "HEARTBEAT_ACCEPTED")
        self.assertNotIn(self.credentials.credentials, event.detail)
        self.db.commit.assert_called_once()

    def test_heartbeat_rejects_missing_or_invalid_credential(self):
        self.set_agent(None)
        with self.assertRaises(HTTPException) as raised:
            node_agents.report_heartbeat("gpu-01", self.body, None, self.db)
        self.assertEqual(raised.exception.status_code, 401)
        event = self.db.add.call_args.args[0]
        self.assertEqual(event.outcome, "AUTHENTICATION_REJECTED")
        self.assertNotIn(self.credentials.credentials, event.detail)
        self.db.commit.assert_called_once()

    def test_heartbeat_rejects_revoked_credential(self):
        self.set_agent(SimpleNamespace(node_id="gpu-01", revoked_at=datetime(2026, 1, 1)))
        with self.assertRaises(HTTPException) as raised:
            node_agents.report_heartbeat("gpu-01", self.body, self.credentials, self.db)
        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(self.db.add.call_args.args[0].outcome, "AUTHENTICATION_REJECTED")

    def test_heartbeat_rejects_credential_used_for_another_node(self):
        self.set_agent(SimpleNamespace(node_id="gpu-01", revoked_at=None, last_seen_at=None))
        with self.assertRaises(HTTPException) as raised:
            node_agents.report_heartbeat("gpu-02", self.body, self.credentials, self.db)
        self.assertEqual(raised.exception.status_code, 403)
        event = self.db.add.call_args.args[0]
        self.assertEqual(event.outcome, "NODE_MISMATCH")
        self.assertEqual(event.authenticated_node_id, "gpu-01")
        self.assertEqual(event.claimed_node_id, "gpu-02")
        self.db.commit.assert_called_once()



class NodeAgentAdministrationTests(unittest.TestCase):
    def test_revocation_invalidates_pending_enrollment_tokens(self):
        agent = SimpleNamespace(node_id="gpu-01", revoked_at=None)
        agent_query = MagicMock()
        agent_query.filter.return_value = agent_query
        agent_query.with_for_update.return_value = agent_query
        agent_query.first.return_value = agent
        token_query = MagicMock()
        token_query.filter.return_value = token_query
        db = MagicMock()
        db.query.side_effect = [agent_query, token_query]

        with patch.object(node_agents, "_utcnow", return_value=datetime(2026, 1, 1)):
            result = node_agents.revoke_node_agent("gpu-01", db, SimpleNamespace(id=7))

        self.assertIs(result, agent)
        self.assertEqual(agent.revoked_at, datetime(2026, 1, 1))
        token_query.update.assert_called_once()
        self.assertEqual(db.add.call_args.args[0].outcome, "REVOKED")
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(agent)

    def test_revoking_unknown_node_returns_not_found(self):
        query = MagicMock()
        query.filter.return_value = query
        query.with_for_update.return_value = query
        query.first.return_value = None
        db = MagicMock()
        db.query.return_value = query

        with self.assertRaises(HTTPException) as raised:
            node_agents.revoke_node_agent("missing-node", db, SimpleNamespace(id=7))

        self.assertEqual(raised.exception.status_code, 404)
        db.commit.assert_not_called()
        db.add.assert_not_called()

    def test_admin_response_schema_does_not_expose_node_credential_hash(self):
        agent = node_agents.NodeAgent(
            node_id="gpu-01",
            name="GPU 01",
            credential_hash="sensitive-hash",
            enrolled_at=datetime(2026, 1, 1),
        )

        response = node_agents.NodeAgentOut.model_validate(agent)

        self.assertEqual(response.node_id, "gpu-01")
        self.assertNotIn("credential_hash", response.model_dump())
        self.assertNotIn("sensitive-hash", repr(response))



class NodeAgentRouteAuthorizationTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from database import get_db
        from logic.dependencies import get_current_user
        from main import app

        self.app = app
        self.db = MagicMock()
        self.user = SimpleNamespace(id=12, role="STUDENT")
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_student_cannot_issue_list_or_revoke_node_agents(self):
        responses = [
            self.client.post(
                "/node-agents/enrollment-tokens",
                json={"node_id": "gpu-01", "name": "GPU 01"},
            ),
            self.client.get("/node-agents"),
            self.client.post("/node-agents/gpu-01/revoke"),
        ]

        self.assertEqual([response.status_code for response in responses], [403, 403, 403])
        self.db.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
