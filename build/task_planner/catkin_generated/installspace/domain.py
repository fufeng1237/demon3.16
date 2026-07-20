#!/usr/bin/env python3
"""Unified domain model for static planning, execution and reallocation.

This is the single source of truth for new task-planner workflows.  Legacy
models remain available only for backwards-compatible experiments.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskState(str, Enum):
    PENDING = 'pending'
    ASSIGNED = 'assigned'
    TO_PICKUP = 'to_pickup'
    LOADING = 'loading'
    TO_DELIVERY = 'to_delivery'
    UNLOADING = 'unloading'
    AT_TRANSFER = 'at_transfer'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'


class ActionType(str, Enum):
    PICKUP = 'pickup'
    DELIVERY = 'delivery'
    REFUEL = 'refuel'
    TRANSFER_DROP = 'transfer_drop'
    TRANSFER_PICK = 'transfer_pick'


@dataclass
class TransportTask:
    task_id: int
    pickup_node: int
    delivery_node: int
    payload: float
    priority: int = 1
    release_time: float = 0.0
    deadline: float = float('inf')
    status: str = TaskState.PENDING.value
    assigned_ship: int = -1
    cargo_node: Optional[int] = None
    transfer_count: int = 0

    @property
    def is_started(self) -> bool:
        return self.status in {TaskState.TO_PICKUP.value, TaskState.LOADING.value,
                               TaskState.TO_DELIVERY.value, TaskState.UNLOADING.value}

    @property
    def is_frozen(self) -> bool:
        return self.status in {TaskState.LOADING.value, TaskState.TO_DELIVERY.value,
                               TaskState.UNLOADING.value}


@dataclass
class ShipState:
    ship_id: int
    name: str
    max_payload: float
    max_energy: float
    max_speed: float
    energy_per_km: float
    current_node: int
    energy: float
    load: float = 0.0
    health: float = 1.0
    current_phase: str = 'idle'
    task_sequence: List[int] = field(default_factory=list)
    completed_tasks: List[int] = field(default_factory=list)
    current_task_id: int = -1
    total_distance: float = 0.0
    total_time: float = 0.0
    failed: bool = False

    @property
    def remaining_capacity(self) -> float:
        return self.max_payload - self.load

    @property
    def energy_ratio(self) -> float:
        return self.energy / self.max_energy if self.max_energy else 0.0

    @property
    def is_idle(self) -> bool:
        return not self.failed and self.current_phase == 'idle' and self.current_task_id < 0


@dataclass
class RouteAction:
    action: ActionType
    node_id: int
    task_id: int = -1
    service_time: float = 0.0
    road_nodes: List[int] = field(default_factory=list)


@dataclass
class RoutePlan:
    ship_id: int
    version: int
    actions: List[RouteAction] = field(default_factory=list)
    frozen_prefix: int = 0
    created_at: float = 0.0

    def node_sequence(self) -> List[int]:
        """Full road-node sequence; duplicate junction nodes are removed."""
        sequence = []
        for action in self.actions:
            for nid in action.road_nodes or [action.node_id]:
                if not sequence or sequence[-1] != nid:
                    sequence.append(nid)
        return sequence


def validate_plan(plan: RoutePlan, ship: ShipState, tasks: Dict[int, TransportTask]) -> List[str]:
    """Validate precedence and cumulative capacity on a single ship plan."""
    errors, loaded = [], set()
    load = ship.load
    for i, action in enumerate(plan.actions):
        if action.task_id < 0:
            continue
        task = tasks.get(action.task_id)
        if task is None:
            errors.append(f'action {i}: unknown task {action.task_id}')
            continue
        if action.action in (ActionType.PICKUP, ActionType.TRANSFER_PICK):
            if action.task_id in loaded:
                errors.append(f'action {i}: duplicate pickup T{action.task_id}')
            loaded.add(action.task_id); load += task.payload
            if load > ship.max_payload + 1e-6:
                errors.append(f'action {i}: capacity exceeded')
        elif action.action in (ActionType.DELIVERY, ActionType.TRANSFER_DROP):
            if action.task_id not in loaded:
                errors.append(f'action {i}: delivery before pickup T{action.task_id}')
            else:
                loaded.remove(action.task_id); load -= task.payload
    if loaded:
        errors.append(f'unclosed pickups: {sorted(loaded)}')
    return errors
