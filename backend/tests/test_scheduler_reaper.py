"""Time-driven scheduler recovery tests (FR-GPU-073, 077, 079, 085, 115)."""
from datetime import timedelta

import models as orm
from logic.scheduler import reaper
from logic.scheduler.domain import JobState, NodeState, ReleaseReason
from scheduler_pg import SchedulerPostgresCase, requires_postgres


@requires_postgres
class SchedulerReaperTests(SchedulerPostgresCase):
    def setUp(self):
        super().setUp()
        self.owner = self.make_user("reaper-owner")

    def test_stale_heartbeat_marks_node_down_releases_work_and_raises_alert(self):
        node = self.make_node("stale", state=NodeState.AVAILABLE)
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node)
        node.last_heartbeat_at = self.now - timedelta(minutes=10)
        self.db.commit()

        result = reaper.reap(self.db, now=self.now)

        self.assertIn(node.id, result.nodes_down)
        self.assertEqual(result.released[job.id], ReleaseReason.NODE_FAILURE)
        self.db.refresh(node)
        self.db.refresh(job)
        self.db.refresh(allocation)
        self.assertEqual(node.state, NodeState.DOWN)
        self.assertEqual(job.state, JobState.QUEUED)
        self.assertIsNotNone(allocation.released_at)
        alerts = self.db.query(orm.NodeAlert).filter_by(node_id=node.id, cleared_at=None).all()
        self.assertTrue(any(str(alert.kind) in ("NODE_DOWN", "AlertKind.NODE_DOWN") for alert in alerts))

    def test_decommissioned_node_revokes_credential_after_last_allocation_is_released(self):
        node = self.make_node("retiring", state=NodeState.DECOMMISSIONING)
        node.credential_hash = "stored-hash"
        node.drain_mode = "soft"
        node.state_reason = "Replacement"
        self.db.commit()

        result = reaper.reap(self.db, now=self.now)

        self.assertIn(node.id, result.decommissioned)
        self.db.refresh(node)
        self.assertEqual(node.state, NodeState.DECOMMISSIONED)
        self.assertEqual(node.credential_revoked_at, self.now)
        self.assertIsNone(node.drain_mode)

    def test_reaping_twice_does_not_release_the_same_allocation_twice(self):
        node = self.make_node("idempotent", state=NodeState.AVAILABLE)
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node)
        node.last_heartbeat_at = self.now - timedelta(minutes=10)
        self.db.commit()

        first = reaper.reap(self.db, now=self.now)
        second = reaper.reap(self.db, now=self.now)

        self.assertIn(job.id, first.released)
        self.assertNotIn(job.id, second.released)
        self.db.refresh(allocation)
        self.assertIsNotNone(allocation.released_at)
