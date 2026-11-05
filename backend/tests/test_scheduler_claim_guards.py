"""Transactional scheduler claim guard tests (FR-GPU-054 to 056, 186)."""
from decimal import Decimal

from logic.scheduler.claim import ClaimStatus, claim
from logic.scheduler.domain import JobState, NodeState, PriorityClass
from logic.scheduler.scoring import ScoreBreakdown
from scheduler_pg import SchedulerPostgresCase, requires_postgres


@requires_postgres
class SchedulerClaimGuardTests(SchedulerPostgresCase):
    def setUp(self):
        super().setUp()
        self.owner = self.make_user("claim-owner")
        self.breakdown = ScoreBreakdown(0.0, 0.5, 0.5, 0.1, Decimal("0.2"), False)

    def _claim(self, job, node, cls=PriorityClass.P0):
        return claim(self.db, job.id, node.id, cls, self.breakdown, "pass-test", "TEST", now=self.now)

    def test_non_queued_job_cannot_be_claimed(self):
        node = self.make_node("job-unavailable")
        job = self.make_job(self.owner, state=JobState.COMPLETED)

        result = self._claim(job, node)

        self.assertEqual(result.status, ClaimStatus.JOB_UNAVAILABLE)
        self.assertIsNone(result.allocation_id)

    def test_node_that_left_available_state_cannot_be_claimed(self):
        node = self.make_node("node-unavailable", state=NodeState.DRAINING)
        job = self.make_job(self.owner)

        result = self._claim(job, node)

        self.assertEqual(result.status, ClaimStatus.NODE_UNAVAILABLE)
        self.db.refresh(job)
        self.assertEqual(job.state, JobState.QUEUED)

    def test_claim_requires_every_requested_gpu_unit(self):
        node = self.make_node("too-small", units=2)
        job = self.make_job(self.owner, gpus=3)

        result = self._claim(job, node)

        self.assertEqual(result.status, ClaimStatus.LOST_CLAIM_RACE)
        self.assertEqual(result.unit_ids, ())
        self.db.refresh(job)
        self.assertEqual(job.state, JobState.QUEUED)

    def test_owner_queue_guard_prevents_guest_from_claiming_shared_node(self):
        node = self.make_node("owner-guard", donation_mode="shared")
        owner_job = self.make_job(self.owner)
        guest_lab = self.make_lab("guest-claim-lab")
        guest = self.make_user("claim-guest", lab=guest_lab)
        guest_job = self.make_job(guest)
        guest_job.lab_id = guest_lab.id
        self.db.commit()

        result = self._claim(guest_job, node, PriorityClass.P1)

        self.assertEqual(result.status, ClaimStatus.OWNER_CLAIM_PENDING)
        self.db.refresh(owner_job)
        self.db.refresh(guest_job)
        self.assertEqual(owner_job.state, JobState.QUEUED)
        self.assertEqual(guest_job.state, JobState.QUEUED)
