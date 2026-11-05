"""Time-driven scheduler recovery tests (FR-GPU-073, 077, 079, 080, 085, 115, 126)."""
from datetime import timedelta

import models as orm
from logic.scheduler import maintenance, reaper
from logic.scheduler.domain import DrainMode, JobState, NodeState, ReleaseReason
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
        self.assertTrue(any(str(alert.kind) == "NODE_DOWN" for alert in alerts))

    def test_unacknowledged_dispatch_times_out_and_requeues_the_job(self):
        node = self.make_node("dispatch-timeout")
        job = self.make_job(self.owner, state=JobState.SCHEDULED)
        allocation = orm.Allocation(
            job_id=job.id, node_id=node.id, claimed_at=self.now - timedelta(minutes=10),
        )
        self.db.add(allocation)
        self.db.flush()
        unit = self.units_of(node)[0]
        self.db.add(orm.AllocationUnit(allocation_id=allocation.id, unit_id=unit.id))
        unit.current_allocation_id = allocation.id
        self.db.commit()

        result = reaper.reap(self.db, now=self.now)

        self.assertEqual(result.released[job.id], ReleaseReason.DISPATCH_TIMEOUT)
        self.db.refresh(job)
        self.db.refresh(allocation)
        self.assertEqual(job.state, JobState.QUEUED)
        self.assertIsNotNone(allocation.released_at)

    def test_walltime_backstop_requests_agent_stop_without_releasing_live_units(self):
        node = self.make_node("walltime")
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node, started_minutes_ago=62)

        result = reaper.reap(self.db, now=self.now)

        self.assertEqual(result.stops_requested[job.id], ReleaseReason.WALLTIME_EXCEEDED)
        self.db.refresh(allocation)
        self.assertIsNotNone(allocation.stop_requested_at)
        self.assertIsNone(allocation.released_at)

    def test_open_maintenance_window_requests_stop_for_allocated_unit(self):
        node = self.make_node("maintenance-start")
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node)
        unit = next(unit for unit in self.units_of(node) if unit.current_allocation_id == allocation.id)
        starts = self.now + timedelta(minutes=1)
        maintenance.book(
            self.db, [unit.id], starts, starts + timedelta(hours=1),
            "Scheduled repair", confirm=True, now=self.now,
        )
        reap_at = self.now + timedelta(minutes=2)
        node.last_heartbeat_at = reap_at
        self.db.commit()

        result = reaper.reap(self.db, now=reap_at)

        self.assertEqual(result.stops_requested[job.id], ReleaseReason.MAINTENANCE)
        self.db.refresh(allocation)
        self.assertIsNotNone(allocation.stop_requested_at)
        self.assertIsNone(allocation.released_at)

    def test_hard_drain_requests_stop_only_after_grace_period(self):
        node = self.make_node("hard-drain", state=NodeState.DRAINING)
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node)
        node.drain_mode = DrainMode.HARD
        node.state_changed_at = self.now - timedelta(hours=1)
        self.db.commit()

        result = reaper.reap(self.db, now=self.now)

        self.assertEqual(result.stops_requested[job.id], ReleaseReason.EMERGENCY_DRAIN)
        self.db.refresh(allocation)
        self.assertIsNotNone(allocation.stop_requested_at)
        self.assertIsNone(allocation.released_at)

    def test_decommissioned_node_revokes_credential_after_last_allocation_is_released(self):
        node = self.make_node("retiring", state=NodeState.DECOMMISSIONING)
        node.credential_hash = "stored-hash"
        node.drain_mode = DrainMode.SOFT
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
