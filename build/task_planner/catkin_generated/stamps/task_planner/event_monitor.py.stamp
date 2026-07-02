#!/usr/bin/env python3
"""
Event Monitor - 内河船舶事件监测器

持续监测船队状态，检测四类事件并生成偏差向量。

Event Types:
  1. ENERGY_LOW    - 船舶能源不足 (< 20%)
  2. FAULT         - 船舶故障 (健康度 < 0.3)
  3. EARLY_FINISH  - 提前完成全部任务
  4. TASK_DELAYED  - 任务预计延期超 deadline

Output:
  - events: 检测到的事件列表
  - deviation_vector: 每船9维偏差向量 (对标技术方案5.2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import json
import os


# ============================================================
# Data Structures
# ============================================================

class EventType(Enum):
    ENERGY_LOW = "energy_low"
    FAULT = "fault"
    EARLY_FINISH = "early_finish"
    TASK_DELAYED = "task_delayed"


class ReallocScope(Enum):
    LOCAL = "local"     # 仅受影响船的任务
    SHIP = "ship"       # 受影响船及其附近船
    GLOBAL = "global"   # 全舰队重分配


@dataclass
class Event:
    type: EventType
    ship_id: int           # 触发事件的船
    task_id: int = -1      # 触发事件的任务 (for TASK_DELAYED)
    timestamp: float = 0.0
    data: Dict = field(default_factory=dict)  # Extra event-specific data

    def to_dict(self):
        return {
            "type": str(self.type.value),
            "ship_id": int(self.ship_id),
            "task_id": int(self.task_id),
            "timestamp": float(self.timestamp),
            "data": {str(k): float(v) if isinstance(v, (int, float)) else v
                     for k, v in self.data.items()}
        }


@dataclass
class DeviationVector:
    """9-dimensional deviation vector per ship (ref: 技术方案 5.2)."""
    ship_id: int
    energy_dev: float = 0.0       # 能源偏差: actual/expected - 1
    progress_dev: float = 0.0     # 进度偏差: 1 - completed/expected
    health_dev: float = 0.0       # 健康偏差: 1 - health
    capacity_dev: float = 0.0     # 容量偏差: load/max_load - 0.9
    has_unassigned: float = 0.0   # 0/1 全局标志
    has_congestion: float = 0.0   # 0/1
    has_collision: float = 0.0    # 0/1
    energy_shortage: float = 0.0  # 0/1 能源不足 (< 20%)
    task_delayed: float = 0.0     # 0/1 任务延期

    def to_list(self) -> List[float]:
        return [
            self.energy_dev, self.progress_dev, self.health_dev,
            self.capacity_dev, self.has_unassigned, self.has_congestion,
            self.has_collision, self.energy_shortage, self.task_delayed
        ]

    @classmethod
    def from_list(cls, ship_id: int, values: List[float]):
        return cls(
            ship_id=ship_id,
            energy_dev=values[0], progress_dev=values[1], health_dev=values[2],
            capacity_dev=values[3], has_unassigned=values[4], has_congestion=values[5],
            has_collision=values[6], energy_shortage=values[7], task_delayed=values[8]
        )


@dataclass
class ReallocationDecision:
    """Result of a reallocation decision."""
    triggered: bool
    event: Event
    scope: ReallocScope
    action: str              # "refuel", "transfer", "wait", "reassign", "incremental"
    cost_before: float
    cost_after: float
    affected_ships: List[int]
    released_tasks: List[int]
    new_assignments: List[Dict]  # (ship_id, task_id, order_index) after realloc
    rationale: str = ""

    def to_dict(self):
        return {
            "triggered": self.triggered,
            "event": self.event.to_dict(),
            "scope": self.scope.value,
            "action": self.action,
            "cost_before": round(self.cost_before, 2),
            "cost_after": round(self.cost_after, 2),
            "affected_ships": self.affected_ships,
            "released_tasks": self.released_tasks,
            "new_assignments": self.new_assignments,
            "rationale": self.rationale
        }


# ============================================================
# Event Monitor
# ============================================================

class EventMonitor:
    """
    Monitor ship fleet for events requiring reallocation.
    """

    def __init__(self,
                 energy_threshold: float = 0.2,
                 health_threshold: float = 0.3,
                 deadline_margin: float = 1.2,    # ETA > deadline × margin → delayed
                 progress_threshold: float = 0.95  # Progress > threshold → early finish
                 ):
        self.energy_threshold = energy_threshold
        self.health_threshold = health_threshold
        self.deadline_margin = deadline_margin
        self.progress_threshold = progress_threshold
        self.event_history: List[Event] = []

    def compute_deviations(self, ships: List, tasks: List, assignments: Dict,
                            global_has_unassigned: bool = False,
                            global_has_congestion: bool = False,
                            global_has_collision: bool = False
                            ) -> List[DeviationVector]:
        """
        Compute 9-dim deviation vector for each ship.
        (Ref: 技术方案 §5.2)
        """
        task_map = {t.id: t for t in tasks}
        deviations = []

        for ship in ships:
            # dimension 0: energy deviation
            energy_dev = max(0.0, 1.0 - ship.energy_ratio)

            # dimension 1: progress deviation
            total_tasks = len(ship.task_queue)
            finished = sum(1 for tid in ship.task_queue
                          if task_map.get(tid) and task_map[tid].status == "finished")
            expected = max(1, total_tasks)
            progress_dev = max(0.0, 1.0 - finished / expected)

            # dimension 2: health deviation
            health_dev = max(0.0, 1.0 - ship.health)

            # dimension 3: capacity deviation
            capacity_dev = max(0.0, ship.load / ship.max_payload - 0.9)

            # dimension 4-8: global flags
            has_unassigned = 1.0 if global_has_unassigned else 0.0
            has_congestion = 1.0 if global_has_congestion else 0.0
            has_collision = 1.0 if global_has_collision else 0.0
            energy_shortage = 1.0 if ship.energy_ratio < self.energy_threshold else 0.0

            # Check if any assigned task is delayed
            task_delayed = 0.0
            for tid in ship.task_queue:
                if tid in task_map:
                    t = task_map[tid]
                    if hasattr(t, 'deadline') and hasattr(t, 'ETA'):
                        if t.deadline < float('inf') and t.ETA > t.deadline * self.deadline_margin:
                            task_delayed = 1.0
                            break

            dv = DeviationVector(
                ship_id=ship.id,
                energy_dev=energy_dev,
                progress_dev=progress_dev,
                health_dev=health_dev,
                capacity_dev=capacity_dev,
                has_unassigned=has_unassigned,
                has_congestion=has_congestion,
                has_collision=has_collision,
                energy_shortage=energy_shortage,
                task_delayed=task_delayed
            )
            deviations.append(dv)

        return deviations

    def detect_events(self, ships: List, tasks: List,
                       current_time: float = 0.0) -> List[Event]:
        """
        Detect all trigger events from current fleet state.
        """
        events = []
        task_map = {t.id: t for t in tasks}

        for ship in ships:
            # Event 1: Energy low
            if ship.energy_ratio < self.energy_threshold:
                events.append(Event(
                    type=EventType.ENERGY_LOW,
                    ship_id=ship.id,
                    timestamp=current_time,
                    data={
                        "energy_ratio": ship.energy_ratio,
                        "energy": ship.energy,
                        "max_energy": ship.max_energy
                    }
                ))

            # Event 2: Fault
            if ship.health < self.health_threshold:
                events.append(Event(
                    type=EventType.FAULT,
                    ship_id=ship.id,
                    timestamp=current_time,
                    data={
                        "health": ship.health,
                        "has_cargo": ship.load > 0
                    }
                ))

            # Event 3: Early finish
            if (ship.status.value == "idle" and
                    len(ship.task_queue) > 0 and
                    all(task_map.get(tid) and task_map[tid].status.value in ("finished",)
                        for tid in ship.task_queue)):
                events.append(Event(
                    type=EventType.EARLY_FINISH,
                    ship_id=ship.id,
                    timestamp=current_time,
                    data={"remaining_capacity": ship.remaining_capacity}
                ))

            # Event 4: Task delayed
            for tid in ship.task_queue:
                t = task_map.get(tid)
                if t and hasattr(t, 'deadline') and hasattr(t, 'ETA'):
                    if t.deadline < float('inf') and t.ETA > t.deadline * self.deadline_margin:
                        events.append(Event(
                            type=EventType.TASK_DELAYED,
                            ship_id=ship.id,
                            task_id=tid,
                            timestamp=current_time,
                            data={
                                "ETA": t.ETA,
                                "deadline": t.deadline,
                                "priority": t.priority
                            }
                        ))

        self.event_history.extend(events)
        return events


# ============================================================
# Reallocation Decision Engine
# ============================================================

class Reallocator:
    """
    Rule-driven reallocation decision engine.
    Handles four event types with cost-based decision logic.
    (Ref: 技术方案 §5.4)
    """

    def __init__(self,
                 refuel_cost_penalty: float = 0.3,     # 加油绕路代价权重
                 transfer_cost_penalty: float = 0.2,    # 任务移交代价权重
                 deadline_penalty_base: float = 100.0   # 延期基础惩罚
                 ):
        self.refuel_cost_penalty = refuel_cost_penalty
        self.transfer_cost_penalty = transfer_cost_penalty
        self.deadline_penalty_base = deadline_penalty_base

    def decide(self,
                event: Event,
                ships: List,
                tasks: List,
                dist_matrix: np.ndarray,
                gas_station_nodes: List[int],
                port_nodes: List[int],
                assignments: List
                ) -> ReallocationDecision:
        """
        Make reallocation decision for a given event.
        """
        task_map = {t.id: t for t in tasks}
        ship_map = {s.id: s for s in ships}

        if event.type == EventType.ENERGY_LOW:
            return self._handle_energy_low(event, ships, tasks, dist_matrix,
                                           gas_station_nodes, ship_map, task_map, assignments)
        elif event.type == EventType.FAULT:
            return self._handle_fault(event, ships, tasks, dist_matrix,
                                      port_nodes, ship_map, task_map, assignments)
        elif event.type == EventType.EARLY_FINISH:
            return self._handle_early_finish(event, ships, tasks, dist_matrix,
                                              ship_map, task_map, assignments)
        elif event.type == EventType.TASK_DELAYED:
            return self._handle_task_delayed(event, ships, tasks, dist_matrix,
                                              ship_map, task_map, assignments)
        else:
            return ReallocationDecision(
                triggered=False, event=event,
                scope=ReallocScope.LOCAL, action="none",
                cost_before=0, cost_after=0,
                affected_ships=[], released_tasks=[], new_assignments=[]
            )

    def _handle_energy_low(self, event: Event, ships: List, tasks: List,
                            dist_matrix: np.ndarray,
                            gas_station_nodes: List[int],
                            ship_map: Dict, task_map: Dict, assignments: List
                            ) -> ReallocationDecision:
        """
        事件: 能源不足
        方案A: 前往最近加油站补给后继续
        方案B: 释放未开始任务，交由其他船执行
        (Ref: 技术方案 §5.4 事件2)
        """
        ship = ship_map[event.ship_id]
        current_node = ship.current_node

        # ── 方案 A: Refuel ──
        # Find nearest gas station
        min_gs_dist = float('inf')
        nearest_gs = -1
        for gs in gas_station_nodes:
            d = dist_matrix[current_node, gs]
            if d < min_gs_dist:
                min_gs_dist = d
                nearest_gs = gs

        cost_a = min_gs_dist * (1.0 + self.refuel_cost_penalty)

        # ── 方案 B: Transfer unstarted tasks ──
        # Identify unstarted tasks (not yet loading)
        unstarted_tasks = []
        for tid in ship.task_queue:
            t = task_map.get(tid)
            if t and t.status.value in ("waiting", "assigned"):
                unstarted_tasks.append(t)

        cost_b = 0.0
        best_alternatives = []
        for task in unstarted_tasks:
            # Find best alternative ship
            min_cost = float('inf')
            best_ship = -1
            for other_ship in ships:
                if other_ship.id == ship.id:
                    continue
                if task.payload > other_ship.remaining_capacity:
                    continue
                d = dist_matrix[other_ship.current_node, task.pickup_node]
                d += dist_matrix[task.pickup_node, task.delivery_node]
                if d < min_cost:
                    min_cost = d
                    best_ship = other_ship.id

            if min_cost < float('inf'):
                cost_b += min_cost
                best_alternatives.append((task.id, best_ship, min_cost))
            else:
                cost_b += 1e6  # Infeasible

        cost_b = cost_b * (1.0 + self.transfer_cost_penalty)

        # Decision
        if cost_a <= cost_b:
            action = "refuel"
            scope = ReallocScope.LOCAL
            affected = [ship.id]
            released = []
            new_assignments = []
            rationale = (f"Refuel at GS_{nearest_gs} (dist={min_gs_dist/1000:.1f}km) "
                         f"cost_a={cost_a/1000:.1f} <= cost_b={cost_b/1000:.1f}")
        else:
            action = "transfer"
            scope = ReallocScope.LOCAL
            affected = [ship.id] + list(set(alt[1] for alt in best_alternatives))
            released = [alt[0] for alt in best_alternatives]
            new_assignments = [{"ship_id": alt[1], "task_id": alt[0]} for alt in best_alternatives]
            rationale = (f"Transfer {len(released)} tasks: "
                         f"cost_a={cost_a/1000:.1f} > cost_b={cost_b/1000:.1f}")

        return ReallocationDecision(
            triggered=True, event=event,
            scope=scope, action=action,
            cost_before=cost_a + cost_b,
            cost_after=min(cost_a, cost_b),
            affected_ships=affected,
            released_tasks=released,
            new_assignments=new_assignments,
            rationale=rationale
        )

    def _handle_fault(self, event: Event, ships: List, tasks: List,
                       dist_matrix: np.ndarray,
                       port_nodes: List[int],
                       ship_map: Dict, task_map: Dict, assignments: List
                       ) -> ReallocationDecision:
        """
        事件: 船舶故障
        - 未装货 → 释放全部任务 → 重新分配
        - 已装货 → 最近港口卸货 + 其他船接力 (方案A) 或 等待维修 (方案B)
        (Ref: 技术方案 §5.4 事件1)
        """
        ship = ship_map[event.ship_id]

        # Separate tasks: loaded vs unloaded
        unloaded_tasks = []  # Tasks not yet started
        loaded_tasks = []    # Tasks in progress (cargo on board)

        for tid in ship.task_queue:
            t = task_map.get(tid)
            if not t:
                continue
            if t.status.value in ("transporting", "loading"):
                loaded_tasks.append(t)
            else:
                unloaded_tasks.append(t)

        # ── Option A: Nearest port offload → transfer ──
        cost_a = 0.0
        offload_port = -1
        if loaded_tasks:
            # Find nearest port for offloading
            min_port_dist = float('inf')
            for pn in port_nodes:
                d = dist_matrix[ship.current_node, pn]
                if d < min_port_dist:
                    min_port_dist = d
                    offload_port = pn
            cost_a += min_port_dist  # Go to nearest port

            # Other ships take over remaining delivery
            for task in loaded_tasks:
                min_pickup = float('inf')
                best_other = -1
                for other_ship in ships:
                    if other_ship.id == ship.id or other_ship.health < 0.5:
                        continue
                    d = dist_matrix[other_ship.current_node, offload_port]
                    d += dist_matrix[offload_port, task.delivery_node]
                    if d < min_pickup:
                        min_pickup = d
                        best_other = other_ship.id
                cost_a += min_pickup if min_pickup < float('inf') else 1e6

        # Also reassign unloaded tasks
        for task in unloaded_tasks:
            min_cost = float('inf')
            for other_ship in ships:
                if other_ship.id == ship.id or other_ship.health < 0.5:
                    continue
                if task.payload > other_ship.remaining_capacity:
                    continue
                d = dist_matrix[other_ship.current_node, task.pickup_node]
                d += dist_matrix[task.pickup_node, task.delivery_node]
                if d < min_cost:
                    min_cost = d
            cost_a += min_cost if min_cost < float('inf') else 1e6

        # ── Option B: Wait for repair ──
        # Cost = delay × all tasks penalty
        repair_time = 3600  # Assume 1 hour repair
        cost_b = 0.0
        for task in loaded_tasks + unloaded_tasks:
            if hasattr(task, 'deadline') and task.deadline < float('inf'):
                cost_b += self.deadline_penalty_base * task.priority

        affected = [ship.id]
        released = [t.id for t in loaded_tasks + unloaded_tasks]

        if cost_a <= cost_b:
            action = "offload_and_transfer"
            rationale = (f"Offload at Port_{offload_port}, then transfer. "
                         f"cost_a={cost_a/1000:.1f} <= cost_b={cost_b/1000:.1f}")
        else:
            action = "wait_for_repair"
            released = []  # Don't release if waiting
            rationale = (f"Wait for repair. "
                         f"cost_a={cost_a/1000:.1f} > cost_b={cost_b/1000:.1f}")

        # Build new assignments for reallocation
        new_assignments = []
        if action == "offload_and_transfer":
            for task in released:
                # Find best ship
                min_cost = float('inf')
                best_ship = -1
                t = task_map[task] if isinstance(task, int) else task
                tid = task if isinstance(task, int) else task.id
                t_obj = task_map[tid] if isinstance(task, int) else task
                for other_ship in ships:
                    if other_ship.id == ship.id or other_ship.health < 0.5:
                        continue
                    if t_obj.payload > other_ship.remaining_capacity:
                        continue
                    d = dist_matrix[other_ship.current_node, t_obj.pickup_node]
                    d += dist_matrix[t_obj.pickup_node, t_obj.delivery_node]
                    if d < min_cost:
                        min_cost = d
                        best_ship = other_ship.id
                if best_ship >= 0:
                    new_assignments.append({"ship_id": best_ship, "task_id": tid})

        return ReallocationDecision(
            triggered=True, event=event,
            scope=ReallocScope.LOCAL, action=action,
            cost_before=max(cost_a, cost_b),
            cost_after=min(cost_a, cost_b),
            affected_ships=affected,
            released_tasks=released,
            new_assignments=new_assignments,
            rationale=rationale
        )

    def _handle_early_finish(self, event: Event, ships: List, tasks: List,
                              dist_matrix: np.ndarray,
                              ship_map: Dict, task_map: Dict, assignments: List
                              ) -> ReallocationDecision:
        """
        事件: 提前完成任务
        船状态 → Idle，参与剩余未分配任务的竞争
        (Ref: 技术方案 §5.4 事件3)
        """
        ship = ship_map[event.ship_id]

        # Find unassigned tasks
        unassigned = [t for t in tasks if t.status.value in ("waiting",)]

        new_assignments = []
        for task in unassigned:
            if task.payload <= ship.remaining_capacity:
                d = dist_matrix[ship.current_node, task.pickup_node]
                d += dist_matrix[task.pickup_node, task.delivery_node]
                if d < float('inf'):
                    new_assignments.append({
                        "ship_id": ship.id,
                        "task_id": task.id,
                        "estimated_cost": d
                    })

        if new_assignments:
            action = "incremental_assign"
            rationale = (f"Ship_{ship.id} is idle. Can take {len(new_assignments)} "
                         f"unassigned tasks without changing existing assignments.")
        else:
            action = "idle"
            rationale = f"Ship_{ship.id} is idle but no suitable unassigned tasks."

        return ReallocationDecision(
            triggered=len(new_assignments) > 0,
            event=event,
            scope=ReallocScope.SHIP,
            action=action,
            cost_before=0,
            cost_after=0,
            affected_ships=[ship.id],
            released_tasks=[],
            new_assignments=new_assignments,
            rationale=rationale
        )

    def _handle_task_delayed(self, event: Event, ships: List, tasks: List,
                              dist_matrix: np.ndarray,
                              ship_map: Dict, task_map: Dict, assignments: List
                              ) -> ReallocationDecision:
        """
        事件: 任务延期
        方案A: 原船继续（承受延期惩罚）
        方案B: 转移任务至最快可用船
        (Ref: 技术方案 §5.4 事件4)
        """
        ship = ship_map[event.ship_id]
        task = task_map[event.task_id]

        # ── 方案 A: Continue with current ship ──
        remaining_dist = dist_matrix[ship.current_node, task.pickup_node]
        remaining_dist += dist_matrix[task.pickup_node, task.delivery_node]
        cost_a = remaining_dist + task.priority * self.deadline_penalty_base

        # ── 方案 B: Transfer to fastest available ship ──
        cost_b = float('inf')
        fastest_ship = -1
        for other_ship in ships:
            if other_ship.id == ship.id:
                continue
            if task.payload > other_ship.remaining_capacity:
                continue
            d = dist_matrix[other_ship.current_node, task.pickup_node]
            d += dist_matrix[task.pickup_node, task.delivery_node]
            # Prefer faster ships
            time_cost = d / other_ship.max_speed if other_ship.max_speed > 0 else float('inf')
            adjusted = d + time_cost * 10  # Small bias for faster ships
            if adjusted < cost_b:
                cost_b = adjusted
                fastest_ship = other_ship.id

        cost_b = cost_b * (1.0 + self.transfer_cost_penalty)

        if cost_a <= cost_b:
            action = "continue"
            scope = ReallocScope.LOCAL
            released = []
            new_assignments = []
            rationale = (f"Ship_{ship.id} continues T{task.id}. "
                         f"cost_a={cost_a/1000:.1f} <= cost_b={cost_b/1000:.1f}")
        else:
            action = "reassign"
            scope = ReallocScope.LOCAL if len([ship.id]) <= 1 else ReallocScope.SHIP
            released = [task.id]
            new_assignments = [{"ship_id": fastest_ship, "task_id": task.id}]
            rationale = (f"Transfer T{task.id} to Ship_{fastest_ship}. "
                         f"cost_a={cost_a/1000:.1f} > cost_b={cost_b/1000:.1f}")

        return ReallocationDecision(
            triggered=(action == "reassign"),
            event=event,
            scope=scope, action=action,
            cost_before=cost_a,
            cost_after=min(cost_a, cost_b),
            affected_ships=[ship.id, fastest_ship] if fastest_ship >= 0 else [ship.id],
            released_tasks=released,
            new_assignments=new_assignments,
            rationale=rationale
        )


# ============================================================
# Reallocation Output Formatting
# ============================================================

def format_reallocation_output(decision: ReallocationDecision,
                                ships_before: List,
                                ships_after: List,
                                tasks: List) -> str:
    """
    Format reallocation result showing before/after comparison.
    """
    task_map = {t.id: t for t in tasks}
    ship_map_before = {s.id: s for s in ships_before}
    ship_map_after = {s.id: s for s in ships_after}

    lines = []
    lines.append("=" * 60)
    lines.append(f"  REALLOCATION EVENT")
    lines.append("=" * 60)
    lines.append(f"  Time: {decision.event.timestamp:.0f}s")
    lines.append(f"  Event: {decision.event.type.value}")
    lines.append(f"  Decision: {decision.action}")
    lines.append(f"  Scope: {decision.scope.value}")
    lines.append(f"  Rationale: {decision.rationale}")
    lines.append("")

    # Before state
    lines.append("  --- Before Reallocation ---")
    for ship in ships_before:
        tid_list = [f"T{tid}" for tid in ship.execution_order]
        lines.append(f"  {ship.name}: [{', '.join(tid_list)}]")
    lines.append("")

    # After state
    lines.append("  --- After Reallocation ---")
    for ship in ships_after:
        tid_list = [f"T{tid}" for tid in ship.execution_order]
        lines.append(f"  {ship.name}: [{', '.join(tid_list)}]")
    lines.append("")

    # Cost change
    delta = decision.cost_after - decision.cost_before
    lines.append(f"  Cost change: {delta/1000:+.1f} km")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Event monitor and reallocator test")
    parser.add_argument("--road-network", type=str,
                        default="/root/demon3.16/src/task_planner/output/road_network.json")
    parser.add_argument("--assignment", type=str,
                        default="/root/demon3.16/src/task_planner/output/assignment_result.json")
    parser.add_argument("--scenario", type=str, default="energy_low",
                        choices=["energy_low", "fault", "early_finish", "task_delayed"],
                        help="Reallocation scenario to test")
    parser.add_argument("--ship-id", type=int, default=0,
                        help="Ship ID to trigger event on")
    args = parser.parse_args()

    # Load data
    from road_network_builder import RoadNetwork
    from task_assigner import Ship, Task, TaskStatus, ShipStatus, allocate, format_allocation_result

    with open(args.road_network, 'r') as f:
        rn_data = json.load(f)
    network = RoadNetwork.from_dict(rn_data)

    gas_nodes = [nid for nid, node in network.nodes.items() if node.is_gas_station]
    port_nodes = [nid for nid, node in network.nodes.items() if node.is_port]

    # Load existing assignment or create demo
    if os.path.exists(args.assignment):
        with open(args.assignment, 'r') as f:
            result_data = json.load(f)
        ships = [Ship(
            id=s["id"], name=s["name"],
            max_payload=s["max_payload"], max_energy=s["max_energy"],
            max_speed=s["max_speed"], position_x=s["position_x"],
            position_y=s["position_y"], current_node=s["current_node"],
            energy=s["energy"], load=s.get("load", 0),
            health=s.get("health", 1.0),
            energy_per_km=s.get("energy_per_km", 2.5)
        ) for s in result_data["ships"]]
        # Update with result assignments
        for s in ships:
            for a in result_data.get("assignments", []):
                if a["ship_id"] == s.id:
                    if a["task_id"] not in s.execution_order:
                        s.execution_order.append(a["task_id"])
                        s.task_queue.append(a["task_id"])
        tasks = [Task.from_dict(td) for td in result_data.get("tasks", [])]
    else:
        print("No existing assignment found. Run task_assigner.py first.")
        sys.exit(1)

    # Trigger scenario
    monitor = EventMonitor()
    reallocator = Reallocator()

    # Inject event
    if args.scenario == "energy_low":
        for s in ships:
            if s.id == args.ship_id:
                s.energy = s.max_energy * 0.15  # 15% energy
                print(f"\n[SCENARIO] Energy low on {s.name}: {s.energy:.0f}/{s.max_energy:.0f} kWh")
                break
    elif args.scenario == "fault":
        for s in ships:
            if s.id == args.ship_id:
                s.health = 0.2
                print(f"\n[SCENARIO] Fault on {s.name}: health={s.health}")
                break
    elif args.scenario == "task_delayed":
        for s in ships:
            if s.id == args.ship_id and s.execution_order:
                tid = s.execution_order[0]
                for t in tasks:
                    if t.id == tid:
                        t.deadline = 100  # Very tight deadline
                        print(f"\n[SCENARIO] Task {tid} delayed on {s.name}")
                        break
                break

    # Detect events
    events = monitor.detect_events(ships, tasks)
    print(f"Detected {len(events)} events")
    for ev in events:
        print(f"  - {ev.type.value} (ship={ev.ship_id})")

    # Make decisions
    ships_before = deepcopy(ships)
    for ev in events:
        decision = reallocator.decide(
            ev, ships, tasks, network.dist_matrix,
            gas_nodes, port_nodes, []
        )
        print(f"\nReallocation decision for {ev.type.value}:")
        print(f"  Triggered: {decision.triggered}")
        print(f"  Action: {decision.action}")
        print(f"  Scope: {decision.scope.value}")
        print(f"  Rationale: {decision.rationale}")
        print(f"  Released tasks: {decision.released_tasks}")

        # Apply the decision to get "after" state
        if decision.triggered:
            ships_after = deepcopy(ships_before)
            # Apply reallocation actions...
            # (simplified: just print the decision)

            formatted = format_reallocation_output(
                decision, ships_before, ships, tasks
            )
            print(formatted)
