"""Administrative node state and inventory lifecycle tests (FR-GPU-110, 112-115, 168-172)."""
from logic.errors import BadRequestError, ConflictError
from logic.scheduler import nodes as node_ops
from logic.scheduler.domain import NodeState
from scheduler_pg import SchedulerPostgresCase, requires_postgres


@requires_postgres
class SchedulerNodeLifecycleTests(SchedulerPostgresCase):
    def test_registration_validates_inventory_and_starts_pending(self):
        with self.assertRaises(BadRequestError):
            node_ops.register_node(self.db, f"zero-{self.suffix}", self.lab.id, "shared", False, 0, now=self.now)
        with self.assertRaises(BadRequestError):
            node_ops.register_node(self.db, f"reserved-{self.suffix}", self.lab.id, "reserved", True, 2, now=self.now)

        registration = node_ops.register_node(
            self.db, f"new-{self.suffix}", self.lab.id, "shared", False, 3, now=self.now,
        )
        self.assertEqual(registration.node.state, NodeState.PENDING_AGENT)
        self.assertEqual(len(self.units_of(registration.node)), 3)
        self.assertIsNotNone(registration.node.enrolment_token_hash)

    def test_drain_return_requires_a_healthy_heartbeat_before_available(self):
        node = self.make_node("repair")
        node_ops.drain(self.db, node.id, "soft", "Fan replacement", now=self.now)
        self.db.refresh(node)
        self.assertEqual(node.state, NodeState.DRAINING)

        node_ops.return_to_service(self.db, node.id, now=self.now)
        self.db.refresh(node)
        self.assertTrue(node.return_requested)
        heartbeat = node_ops.heartbeat(self.db, node.id, node.unit_count, [], now=self.now)
        self.assertTrue(heartbeat.became_available)
        self.assertEqual(heartbeat.state, NodeState.AVAILABLE)

    def test_decommission_can_be_aborted_but_terminal_nodes_cannot_return(self):
        node = self.make_node("abort")
        node_ops.decommission(self.db, node.id, "soft", "Retiring", now=self.now)
        node_ops.abort_decommission(self.db, node.id, "Replacement postponed", now=self.now)
        self.db.refresh(node)
        self.assertEqual(node.state, NodeState.DRAINING)

        retired = self.make_node("retired", state=NodeState.DECOMMISSIONED)
        with self.assertRaises(ConflictError):
            node_ops.return_to_service(self.db, retired.id, now=self.now)

    def test_unit_count_can_grow_and_retire_only_free_units(self):
        node = self.make_node("inventory", units=2)
        node_ops.drain(self.db, node.id, "soft", "Inventory adjustment", now=self.now)
        node_ops.confirm_unit_count(self.db, node.id, 3, now=self.now)
        self.db.refresh(node)
        self.assertEqual(node.unit_count, 3)
        self.assertEqual(len([u for u in self.units_of(node) if u.retired_at is None]), 3)

        node_ops.confirm_unit_count(self.db, node.id, 1, now=self.now)
        self.db.refresh(node)
        self.assertEqual(node.unit_count, 1)
        self.assertEqual(len([u for u in self.units_of(node) if u.retired_at is None]), 1)
