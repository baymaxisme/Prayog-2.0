"""GPU maintenance preview, collision confirmation, move and cancellation tests."""
from datetime import timedelta

from logic.errors import BadRequestError
from logic.scheduler import maintenance
from scheduler_pg import SchedulerPostgresCase, requires_postgres


@requires_postgres
class SchedulerMaintenanceTests(SchedulerPostgresCase):
    def setUp(self):
        super().setUp()
        self.owner = self.make_user("maintenance-owner")

    def _running_unit(self):
        node = self.make_node("maintenance-node")
        job = self.make_job(self.owner)
        allocation = self.make_running(job, node)
        unit = next(unit for unit in self.units_of(node) if unit.current_allocation_id == allocation.id)
        return job, unit

    def test_preview_reports_collision_and_book_requires_explicit_confirmation(self):
        job, unit = self._running_unit()
        starts = self.now + timedelta(minutes=5)
        ends = self.now + timedelta(hours=2)

        preview = maintenance.preview(self.db, [unit.id], starts, now=self.now)
        self.assertEqual([collision.job_id for collision in preview.collisions], [job.id])

        with self.assertRaises(maintenance.CollisionNotConfirmed):
            maintenance.book(self.db, [unit.id], starts, ends, "GPU repair", now=self.now)
        self.assertFalse(any(unit.id in booking.unit_ids for booking in maintenance.bookings(self.db, now=self.now)))

    def test_confirmed_booking_can_be_listed_and_cancelled(self):
        _, unit = self._running_unit()
        starts = self.now + timedelta(minutes=5)
        ends = self.now + timedelta(hours=2)

        booking_id, preview = maintenance.book(
            self.db, [unit.id], starts, ends, "GPU repair", confirm=True, now=self.now,
        )
        self.assertTrue(preview.collisions)
        matches = [booking for booking in maintenance.bookings(self.db, now=self.now)
                   if booking.booking_id == booking_id]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].unit_ids, (unit.id,))

        maintenance.cancel(self.db, booking_id)
        self.assertFalse(any(booking.booking_id == booking_id
                             for booking in maintenance.bookings(self.db, now=self.now)))

    def test_invalid_window_and_unknown_unit_are_rejected(self):
        _, unit = self._running_unit()
        with self.assertRaises(BadRequestError):
            maintenance.book(self.db, [unit.id], self.now, self.now, "invalid", now=self.now)
        with self.assertRaises(BadRequestError):
            maintenance.preview(self.db, [999999999], self.now + timedelta(hours=1), now=self.now)
