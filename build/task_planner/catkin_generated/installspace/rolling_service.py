#!/usr/bin/env python3
"""Event-driven rolling coordinator for the unified task-planner pipeline."""
from typing import Dict, List
from domain import TaskState
from execution_engine import ExecutionEngine, PlannerEvent


class RollingPlanningService:
    def __init__(self, planner, engine: ExecutionEngine, replan_interval=900.0,
                 energy_threshold=0.20):
        self.planner, self.engine = planner, engine
        self.replan_interval = replan_interval
        self.energy_threshold = energy_threshold
        self.next_replan = 0.0
        self.plan_history = []

    def replan(self, reason='initial'):
        # A ship already executing a plan keeps its remaining suffix.  It is
        # intentionally excluded from this replan, so its reserved tasks
        # cannot be duplicated onto another ship.
        reserved = set()
        for sid, plan in self.engine.plans.items():
            ship = self.engine.ships.get(sid)
            if ship and ship.current_task_id >= 0:
                cursor = self.engine.cursors.get(sid)
                start = cursor.action_index if cursor else 0
                reserved.update(a.task_id for a in plan.actions[start:] if a.task_id >= 0)
        mutable = {tid for tid, task in self.engine.tasks.items()
                   if task.status in (TaskState.PENDING.value, TaskState.ASSIGNED.value,
                                      TaskState.AT_TRANSFER.value)
                   and task.release_time <= self.engine.time and tid not in reserved}
        plans = self.planner.plan(self.engine.ships, self.engine.tasks, self.engine.time, mutable)
        self.engine.set_plans(plans)
        self.plan_history.append({'time': self.engine.time, 'reason': reason,
                                  'version': self.planner.version,
                                  'mutable_tasks': sorted(mutable)})
        self.next_replan = self.engine.time + self.replan_interval
        return plans

    def step(self, dt: float):
        before = len(self.engine.events)
        self.engine.step(dt)
        new_events = self.engine.events[before:]
        event_replan = any(e.kind in ('ship_fault', 'cargo_transfer_required', 'task_released') for e in new_events)
        energy_replan = any((not s.failed and s.energy_ratio < self.energy_threshold)
                            for s in self.engine.ships.values())
        if event_replan or energy_replan or self.engine.time >= self.next_replan:
            self.replan('event' if event_replan else ('low_energy' if energy_replan else 'periodic'))
        return new_events

    def inject_fault(self, ship_id: int, reason='fault'):
        self.engine.fail_ship(ship_id, reason)
        return self.replan('ship_fault')
