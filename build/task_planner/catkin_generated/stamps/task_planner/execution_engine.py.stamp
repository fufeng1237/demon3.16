#!/usr/bin/env python3
"""Time-accurate plan executor with task freezing and safe cargo transfer."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from domain import ActionType, RoutePlan, ShipState, TaskState, TransportTask
from route_builder import shortest_path


@dataclass
class PlannerEvent:
    kind: str
    ship_id: int = -1
    task_id: int = -1
    time: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class _ExecutionCursor:
    action_index: int = 0
    edge_index: int = 0
    edge_remaining: float = 0.0
    service_remaining: float = 0.0


class ExecutionEngine:
    """Executes RoutePlan actions in elapsed time; never reassigns frozen cargo."""
    def __init__(self, rn, ships: Dict[int, ShipState], tasks: Dict[int, TransportTask], port_ids: List[int]):
        self.rn, self.ships, self.tasks, self.port_ids = rn, ships, tasks, port_ids
        self.time = 0.0
        self.plans: Dict[int, RoutePlan] = {}
        self.cursors: Dict[int, _ExecutionCursor] = {}
        self.events: List[PlannerEvent] = []

    def set_plans(self, plans: Dict[int, RoutePlan]):
        """Replace plans only for idle ships; active actions are deliberately preserved."""
        for sid, plan in plans.items():
            ship = self.ships.get(sid)
            if ship is None or ship.failed or ship.current_task_id >= 0:
                continue
            self.plans[sid] = plan
            self.cursors[sid] = _ExecutionCursor()
            ship.task_sequence = [a.task_id for a in plan.actions if a.action == ActionType.PICKUP]
            for action in plan.actions:
                if action.action == ActionType.PICKUP and action.task_id in self.tasks:
                    task = self.tasks[action.task_id]
                    if task.status in (TaskState.PENDING.value, TaskState.AT_TRANSFER.value, TaskState.ASSIGNED.value):
                        task.status = TaskState.ASSIGNED.value
                        task.assigned_ship = sid

    def fail_ship(self, sid: int, reason='fault'):
        ship = self.ships.get(sid)
        if not ship or ship.failed:
            return
        ship.failed = True; ship.health = 0.0
        tid = ship.current_task_id
        if tid >= 0 and self.tasks[tid].status in (TaskState.TO_DELIVERY.value, TaskState.UNLOADING.value):
            task = self.tasks[tid]
            depot = min(self.port_ids, key=lambda n: self.rn.dist_matrix[ship.current_node, n])
            task.pickup_node = depot; task.cargo_node = depot; task.status = TaskState.AT_TRANSFER.value
            task.assigned_ship = -1; task.transfer_count += 1
            ship.load = max(0.0, ship.load - task.payload)
            self.events.append(PlannerEvent('cargo_transfer_required', sid, tid, self.time,
                                            {'drop_node': depot, 'reason': reason}))
        elif tid >= 0:
            task = self.tasks[tid]; task.status = TaskState.PENDING.value; task.assigned_ship = -1
            self.events.append(PlannerEvent('task_released', sid, tid, self.time, {'reason': reason}))
        ship.current_task_id = -1; ship.current_phase = 'failed'; self.plans.pop(sid, None); self.cursors.pop(sid, None)
        self.events.append(PlannerEvent('ship_fault', sid, time=self.time, details={'reason': reason}))

    def step(self, dt: float):
        for sid, ship in self.ships.items():
            if not ship.failed:
                self._advance_ship(sid, dt)
        self.time += dt

    def _advance_ship(self, sid: int, dt: float):
        ship, plan = self.ships[sid], self.plans.get(sid)
        cursor = self.cursors.get(sid)
        if not plan or not cursor or cursor.action_index >= len(plan.actions):
            ship.current_phase = 'idle'; ship.current_task_id = -1; return
        remaining = dt
        while remaining > 1e-9 and cursor.action_index < len(plan.actions):
            action = plan.actions[cursor.action_index]
            if cursor.service_remaining > 0:
                use = min(remaining, cursor.service_remaining); cursor.service_remaining -= use; remaining -= use; ship.total_time += use
                if cursor.service_remaining > 1e-9: break
                self._complete_action(ship, action); cursor.action_index += 1; cursor.edge_index = 0; continue
            path = action.road_nodes or shortest_path(self.rn, ship.current_node, action.node_id)
            if cursor.edge_index >= len(path) - 1:
                cursor.service_remaining = action.service_time
                if action.action == ActionType.PICKUP:
                    ship.current_phase = TaskState.LOADING.value
                elif action.action == ActionType.DELIVERY:
                    ship.current_phase = TaskState.UNLOADING.value
                elif action.action == ActionType.REFUEL:
                    ship.current_phase = 'refueling'
                continue
            u, v = path[cursor.edge_index], path[cursor.edge_index + 1]
            if cursor.edge_remaining <= 0:
                cursor.edge_remaining = float(self.rn.dist_matrix[u, v])
            max_distance = ship.max_speed * remaining
            travel = min(max_distance, cursor.edge_remaining)
            use = travel / max(ship.max_speed, 1e-6)
            ship.total_distance += travel; ship.total_time += use
            ship.energy -= travel / 1000.0 * ship.energy_per_km
            cursor.edge_remaining -= travel; remaining -= use; ship.current_phase = 'sailing'
            if cursor.edge_remaining <= 1e-6:
                ship.current_node = v; cursor.edge_index += 1; cursor.edge_remaining = 0.0

    def _complete_action(self, ship: ShipState, action):
        task = self.tasks.get(action.task_id)
        if action.action == ActionType.REFUEL:
            ship.energy = ship.max_energy
            self.events.append(PlannerEvent('refueled', ship.ship_id, time=self.time,
                                            details={'node': action.node_id}))
        elif action.action == ActionType.PICKUP and task:
            ship.load += task.payload; ship.current_task_id = task.task_id
            task.assigned_ship = ship.ship_id; task.status = TaskState.TO_DELIVERY.value
        elif action.action == ActionType.DELIVERY and task:
            ship.load = max(0.0, ship.load - task.payload); ship.completed_tasks.append(task.task_id)
            task.status = TaskState.COMPLETED.value; task.assigned_ship = ship.ship_id
            ship.current_task_id = -1; self.events.append(PlannerEvent('task_completed', ship.ship_id, task.task_id, self.time))
