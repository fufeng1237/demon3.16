#!/usr/bin/env python3
"""
Task Assigner - 内河船舶任务分配器

基于路网距离的贪心 + 2-opt 优化分配算法。

Method:
  1. 构建距离代价矩阵 (船→装货港→卸货港)
  2. 贪心初始分配 (按优先级×1/deadline 排序)
  3. 2-opt 局部优化 (船内任务顺序优化)
  4. 任务转移优化 (船间任务转移)
  5. 输出每艘船的决策序列

Output format:
  Ship_i 决策:
    Task sequence: T3 → T1 → T7
    Step 1: Navigate pos → Port_A (load, 500t)
    Step 2: Navigate Port_A → Port_C (unload, 500t)
    ...
    Estimated total distance: 45.2 km
    Estimated total time: 3.5 h
"""

import numpy as np
import json
import yaml
import os
import sys
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# ============================================================
# Data Structures
# ============================================================

class ShipStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    LOADING = "loading"
    UNLOADING = "unloading"
    REFUELING = "refueling"
    WAITING = "waiting"
    EMERGENCY = "emergency"


class TaskStatus(Enum):
    WAITING = "waiting"
    ASSIGNED = "assigned"
    LOADING = "loading"
    TRANSPORTING = "transporting"
    FINISHED = "finished"


@dataclass
class Task:
    id: int
    pickup_node: int       # 装货港 RoadNode ID
    delivery_node: int     # 卸货港 RoadNode ID
    payload: float         # 货物重量 (吨)
    start_time: float      # 最早开始时间
    deadline: float        # 截止时间
    priority: int          # 1=低, 2=中, 3=高
    status: TaskStatus = TaskStatus.WAITING
    assigned_ship: int = -1

    def to_dict(self):
        return {
            "id": int(self.id),
            "pickup_node": int(self.pickup_node),
            "delivery_node": int(self.delivery_node),
            "payload": float(self.payload),
            "start_time": float(self.start_time),
            "deadline": float(self.deadline),
            "priority": int(self.priority),
            "status": str(self.status.value),
            "assigned_ship": int(self.assigned_ship)
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            pickup_node=data["pickup_node"],
            delivery_node=data["delivery_node"],
            payload=data["payload"],
            start_time=data.get("start_time", 0),
            deadline=data.get("deadline", float('inf')),
            priority=data.get("priority", 1),
            status=TaskStatus(data.get("status", "waiting")),
            assigned_ship=data.get("assigned_ship", -1)
        )


@dataclass
class Ship:
    id: int
    name: str
    max_payload: float      # 最大载重 (吨)
    max_energy: float       # 最大能源 (kWh)
    max_speed: float        # 最大航速 (m/s)
    position_x: float       # 当前位置 X
    position_y: float       # 当前位置 Y
    current_node: int       # 最近的路网节点 ID
    energy: float           # 当前能源
    load: float = 0.0       # 当前载货量
    health: float = 1.0     # 健康状态 0~1
    status: ShipStatus = ShipStatus.IDLE
    task_queue: List[int] = field(default_factory=list)          # 已分配任务ID
    execution_order: List[int] = field(default_factory=list)     # 执行顺序
    energy_per_km: float = 2.5    # 每公里能耗 (kWh/km)
    ETA: float = 0.0              # 预计完成时间 (秒)

    def to_dict(self):
        return {
            "id": int(self.id),
            "name": str(self.name),
            "max_payload": float(self.max_payload),
            "max_energy": float(self.max_energy),
            "max_speed": float(self.max_speed),
            "position_x": float(self.position_x),
            "position_y": float(self.position_y),
            "current_node": int(self.current_node),
            "energy": float(self.energy),
            "load": float(self.load),
            "health": float(self.health),
            "status": str(self.status.value),
            "task_queue": [int(x) for x in self.task_queue],
            "execution_order": [int(x) for x in self.execution_order],
            "energy_per_km": float(self.energy_per_km),
            "ETA": float(self.ETA)
        }

    @property
    def remaining_capacity(self) -> float:
        return self.max_payload - self.load

    @property
    def energy_ratio(self) -> float:
        return self.energy / self.max_energy if self.max_energy > 0 else 0


@dataclass
class Assignment:
    """Represents a task assigned to a ship with execution order."""
    ship_id: int
    task_id: int
    order_index: int       # Position in execution sequence
    estimated_cost: float  # Estimated additional cost

    def to_dict(self):
        return {
            "ship_id": int(self.ship_id),
            "task_id": int(self.task_id),
            "order_index": int(self.order_index),
            "estimated_cost": round(float(self.estimated_cost), 2)
        }


@dataclass
class AllocationResult:
    """Complete allocation result for all ships."""
    assignments: List[Assignment]
    ships: List[Ship]
    tasks: List[Task]
    total_distance: float      # Fleet total distance (km)
    total_energy: float        # Fleet total energy (kWh)
    completed_tasks: int
    unassigned_tasks: int
    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        import numpy as np

        def safe_val(v):
            """Convert numpy types to Python native types."""
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            return v

        return {
            "assignments": [a.to_dict() for a in self.assignments],
            "ships": [s.to_dict() for s in self.ships],
            "tasks": [t.to_dict() for t in self.tasks],
            "total_distance": round(safe_val(self.total_distance), 2),
            "total_energy": round(safe_val(self.total_energy), 2),
            "completed_tasks": int(safe_val(self.completed_tasks)),
            "unassigned_tasks": int(safe_val(self.unassigned_tasks)),
            "cost_breakdown": {str(k): round(safe_val(v), 2)
                               for k, v in self.cost_breakdown.items()}
        }


# ============================================================
# Cost Functions
# ============================================================

def compute_task_path_cost(dist_matrix: np.ndarray,
                            ship_node: int,
                            task: Task) -> float:
    """
    Compute total distance for a ship to execute a task:
    ship_pos → pickup_node → delivery_node
    """
    d1 = dist_matrix[ship_node, task.pickup_node]
    d2 = dist_matrix[task.pickup_node, task.delivery_node]
    if d1 == np.inf or d2 == np.inf:
        return np.inf
    return d1 + d2


def compute_sequence_cost(dist_matrix: np.ndarray,
                           ship_node: int,
                           task_sequence: List[Task]) -> float:
    """
    Compute total distance for executing a sequence of tasks.
    ship → T0.pickup → T0.delivery → T1.pickup → T1.delivery → ...
    """
    if not task_sequence:
        return 0.0

    total = 0.0
    current_node = ship_node

    for task in task_sequence:
        d1 = dist_matrix[current_node, task.pickup_node]
        d2 = dist_matrix[task.pickup_node, task.delivery_node]
        if d1 == np.inf or d2 == np.inf:
            return np.inf
        total += d1 + d2
        current_node = task.delivery_node

    return total


def compute_energy_cost(distance_km: float, energy_per_km: float) -> float:
    """Compute energy consumption for a given distance."""
    return distance_km * energy_per_km


def compute_time_cost(distance_m: float, speed_ms: float) -> float:
    """Compute time in seconds for a given distance."""
    return distance_m / speed_ms if speed_ms > 0 else float('inf')


# ============================================================
# Greedy Initial Assignment
# ============================================================

def greedy_assign(ships: List[Ship],
                   tasks: List[Task],
                   dist_matrix: np.ndarray,
                   node_id_to_name: Dict[int, str] = None
                   ) -> AllocationResult:
    """
    Greedy task assignment.
    Sort tasks by priority/urgency, assign each to the ship with minimum incremental cost.
    """
    ships = deepcopy(ships)
    tasks = deepcopy(tasks)
    assignments = []

    # Sort tasks: high priority × (1 / deadline) first
    def urgency(task: Task) -> float:
        urgency = task.priority * 10.0
        if task.deadline < float('inf'):
            urgency += 100.0 / max(task.deadline, 1.0)
        return -urgency  # negative for descending sort

    unassigned = [t for t in tasks if t.status != TaskStatus.FINISHED]
    unassigned.sort(key=urgency)

    for task in unassigned:
        best_ship_idx = -1
        best_cost = float('inf')

        for i, ship in enumerate(ships):
            # Capacity check
            if task.payload > ship.remaining_capacity:
                continue

            # Energy check: can the ship reach pickup + delivery?
            cost = compute_task_path_cost(dist_matrix, ship.current_node, task)
            if cost == np.inf:
                continue

            energy_needed = compute_energy_cost(cost / 1000.0, ship.energy_per_km)
            if energy_needed > ship.energy * 0.8:  # Safety margin
                continue

            if cost < best_cost:
                best_cost = cost
                best_ship_idx = i

        if best_ship_idx >= 0:
            task.status = TaskStatus.ASSIGNED
            task.assigned_ship = ships[best_ship_idx].id
            ships[best_ship_idx].load += task.payload
            ships[best_ship_idx].task_queue.append(task.id)
            # Append to execution order (will be optimized later)
            ships[best_ship_idx].execution_order.append(task.id)
            idx = len(ships[best_ship_idx].execution_order) - 1
            assignments.append(Assignment(
                ship_id=ships[best_ship_idx].id,
                task_id=task.id,
                order_index=idx,
                estimated_cost=best_cost
            ))

    # Calculate total metrics
    total_distance = 0.0
    total_energy = 0.0
    for ship in ships:
        ship_tasks = [t for t in tasks if t.id in ship.execution_order]
        seq_dist = compute_sequence_cost(dist_matrix, ship.current_node, ship_tasks)
        if seq_dist != np.inf:
            total_distance += seq_dist / 1000.0  # Convert to km
        total_energy += compute_energy_cost(total_distance, ship.energy_per_km)

    completed = sum(1 for t in tasks if t.status == TaskStatus.FINISHED)
    unassigned_count = sum(1 for t in tasks if t.status == TaskStatus.WAITING)

    result = AllocationResult(
        assignments=assignments,
        ships=ships,
        tasks=tasks,
        total_distance=total_distance,
        total_energy=total_energy,
        completed_tasks=completed,
        unassigned_tasks=unassigned_count,
        cost_breakdown={"distance_km": total_distance, "energy_kwh": total_energy}
    )

    return result


# ============================================================
# 2-opt Local Optimization (Within Ship)
# ============================================================

def two_opt_optimize(ship: Ship,
                      tasks: List[Task],
                      dist_matrix: np.ndarray,
                      max_iterations: int = 100) -> List[int]:
    """
    2-opt optimization for a single ship's task execution sequence.
    Repeatedly swap two tasks in the sequence if it reduces total path length.
    """
    if len(ship.execution_order) < 2:
        return ship.execution_order

    # Map task_id → Task
    task_map = {t.id: t for t in tasks}
    order = list(ship.execution_order)

    def total_path_cost(seq: List[int]) -> float:
        current_node = ship.current_node
        total = 0.0
        for tid in seq:
            t = task_map[tid]
            total += dist_matrix[current_node, t.pickup_node]
            total += dist_matrix[t.pickup_node, t.delivery_node]
            current_node = t.delivery_node
        return total

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        best_cost = total_path_cost(order)

        for i in range(len(order)):
            for j in range(i + 2, len(order)):
                # Try swapping tasks at positions i and j
                new_order = order.copy()
                # Swap: reverse the segment between i and j (2-opt)
                new_order[i:j+1] = reversed(new_order[i:j+1])

                new_cost = total_path_cost(new_order)
                if new_cost < best_cost - 1e-6:  # Small tolerance
                    order = new_order
                    best_cost = new_cost
                    improved = True

    return order


# ============================================================
# Inter-Ship Task Transfer Optimization
# ============================================================

def optimize_task_transfer(ships: List[Ship],
                            tasks: List[Task],
                            dist_matrix: np.ndarray,
                            max_iterations: int = 50) -> Tuple[List[Ship], List[Assignment]]:
    """
    Try transferring tasks between ships to reduce total fleet cost.
    """
    ships = deepcopy(ships)
    tasks = deepcopy(tasks)
    task_map = {t.id: t for t in tasks}

    def fleet_total_cost(ships: List[Ship]) -> float:
        total = 0.0
        for ship in ships:
            ship_tasks = [task_map[tid] for tid in ship.execution_order if tid in task_map]
            cost = compute_sequence_cost(dist_matrix, ship.current_node, ship_tasks)
            if cost != np.inf:
                total += cost
        return total

    improved = True
    iteration = 0
    assignments = []

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        best_cost = fleet_total_cost(ships)

        for i, ship_i in enumerate(ships):
            if not ship_i.execution_order:
                continue
            for j, ship_j in enumerate(ships):
                if i == j:
                    continue
                for ti_idx, task_id in enumerate(ship_i.execution_order):
                    task = task_map[task_id]

                    # Check capacity of ship_j
                    if task.payload > ship_j.remaining_capacity:
                        continue

                    # Try transfer: remove from ship_i, add to ship_j
                    ship_i.execution_order.remove(task_id)
                    ship_i.task_queue.remove(task_id)

                    # Find best insertion position in ship_j
                    best_pos = len(ship_j.execution_order)
                    best_new_cost = float('inf')
                    for pos in range(len(ship_j.execution_order) + 1):
                        new_order = ship_j.execution_order.copy()
                        new_order.insert(pos, task_id)
                        ship_tasks = [task_map[tid] for tid in new_order if tid in task_map]
                        cost = compute_sequence_cost(dist_matrix, ship_j.current_node, ship_tasks)
                        if cost < best_new_cost:
                            best_new_cost = cost
                            best_pos = pos

                    # Calculate new fleet cost
                    ship_j.execution_order.insert(best_pos, task_id)
                    ship_j.task_queue.append(task_id)
                    # Temporarily update loads
                    ship_i.load -= task.payload
                    ship_j.load += task.payload

                    new_cost = fleet_total_cost(ships)

                    if new_cost < best_cost - 1e-6:
                        best_cost = new_cost
                        improved = True
                        task.assigned_ship = ship_j.id
                        # Keep the change
                    else:
                        # Revert
                        ship_j.execution_order.remove(task_id)
                        ship_j.task_queue.remove(task_id)
                        ship_i.execution_order.insert(ti_idx, task_id)
                        ship_i.task_queue.append(task_id)
                        ship_i.load += task.payload
                        ship_j.load -= task.payload

    # Rebuild assignments
    for ship in ships:
        for idx, tid in enumerate(ship.execution_order):
            task = task_map[tid]
            cost = compute_task_path_cost(dist_matrix, ship.current_node, task)
            assignments.append(Assignment(
                ship_id=ship.id,
                task_id=tid,
                order_index=idx,
                estimated_cost=cost
            ))

    return ships, assignments


# ============================================================
# Full Allocation Pipeline
# ============================================================

def allocate(ships: List[Ship],
              tasks: List[Task],
              dist_matrix: np.ndarray,
              enable_2opt: bool = True,
              enable_transfer: bool = True,
              node_id_to_name: Dict[int, str] = None
              ) -> AllocationResult:
    """
    Full allocation pipeline:
    1. Greedy initial assignment
    2. 2-opt optimization (intra-ship)
    3. Task transfer optimization (inter-ship)
    """
    ships = deepcopy(ships)
    tasks = deepcopy(tasks)

    # Step 1: Greedy assignment
    result = greedy_assign(ships, tasks, dist_matrix, node_id_to_name)

    # Step 2: 2-opt per ship
    if enable_2opt:
        for ship in result.ships:
            ship.execution_order = two_opt_optimize(
                ship, result.tasks, dist_matrix
            )

    # Step 3: Inter-ship transfer
    if enable_transfer:
        result.ships, result.assignments = optimize_task_transfer(
            result.ships, result.tasks, dist_matrix
        )

    # Recalculate totals
    task_map = {t.id: t for t in result.tasks}
    total_distance = 0.0
    total_energy = 0.0
    for ship in result.ships:
        ship_tasks = [task_map[tid] for tid in ship.execution_order if tid in task_map]
        seq_dist = compute_sequence_cost(dist_matrix, ship.current_node, ship_tasks)
        if seq_dist != np.inf:
            total_distance += seq_dist / 1000.0  # km
        total_energy += compute_energy_cost(total_distance, ship.energy_per_km)
        # Update ETA
        if ship.max_speed > 0:
            ship.ETA = compute_time_cost(seq_dist, ship.max_speed)

    result.total_distance = total_distance
    result.total_energy = total_energy
    result.completed_tasks = sum(1 for t in result.tasks if t.status == TaskStatus.FINISHED)
    result.unassigned_tasks = sum(1 for t in result.tasks if t.status == TaskStatus.WAITING)
    result.cost_breakdown = {"distance_km": total_distance, "energy_kwh": total_energy}

    return result


# ============================================================
# Decision Output Formatting
# ============================================================

def format_ship_decision(ship: Ship,
                          tasks: List[Task],
                          dist_matrix: np.ndarray,
                          node_info: Dict[int, str] = None) -> str:
    """
    Format the decision output for a single ship.
    """
    task_map = {t.id: t for t in tasks}
    lines = []
    lines.append(f"{'─'*60}")
    lines.append(f"{ship.name} (capacity: {ship.max_payload}t, "
                 f"energy: {ship.energy:.0f}/{ship.max_energy:.0f}kWh)")

    if not ship.execution_order:
        lines.append("  No tasks assigned.")
        return "\n".join(lines)

    task_str = " → ".join(f"T{tid}" for tid in ship.execution_order)
    lines.append(f"  Task sequence: {task_str}")

    current_node = ship.current_node
    total_dist = 0.0
    step_num = 1

    for tid in ship.execution_order:
        task = task_map[tid]

        # Navigate to pickup
        d1 = float(dist_matrix[current_node, task.pickup_node])
        if d1 == float('inf') or d1 <= 0:
            d1 = 0.0  # Ship already at pickup
        total_dist += d1

        pickup_name = node_info.get(task.pickup_node, f"Node_{task.pickup_node}") if node_info else f"Node_{task.pickup_node}"
        delivery_name = node_info.get(task.delivery_node, f"Node_{task.delivery_node}") if node_info else f"Node_{task.delivery_node}"

        lines.append(f"  Step {step_num}: Navigate → {pickup_name} "
                     f"(load, {task.payload}t, dist={d1/1000:.1f}km)")
        step_num += 1

        # Transport to delivery
        d2 = float(dist_matrix[task.pickup_node, task.delivery_node])
        if d2 == float('inf'):
            d2 = 0.0
        total_dist += d2
        lines.append(f"  Step {step_num}: Navigate → {delivery_name} "
                     f"(unload, {task.payload}t, dist={d2/1000:.1f}km)")
        step_num += 1

        current_node = task.delivery_node

    energy = compute_energy_cost(total_dist / 1000.0, ship.energy_per_km)
    time_h = compute_time_cost(total_dist, ship.max_speed) / 3600.0 if ship.max_speed > 0 else 0

    lines.append(f"  ──────────────────────────────")
    lines.append(f"  Estimated total distance: {total_dist/1000:.1f} km")
    lines.append(f"  Estimated total energy: {energy:.1f} kWh")
    lines.append(f"  Estimated total time: {time_h:.1f} h")
    lines.append(f"  Remaining capacity: {ship.remaining_capacity:.0f} t")

    return "\n".join(lines)


def format_allocation_result(result: AllocationResult,
                               dist_matrix: np.ndarray,
                               node_info: Dict[int, str] = None,
                               scenario_name: str = "Task Allocation") -> str:
    """
    Format the complete allocation result as a readable string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {scenario_name}")
    lines.append("=" * 60)

    for ship in result.ships:
        lines.append(format_ship_decision(ship, result.tasks, dist_matrix, node_info))
        lines.append("")

    lines.append("─" * 60)
    lines.append("Fleet Summary:")
    lines.append(f"  Total distance: {result.total_distance:.1f} km")
    lines.append(f"  Total energy: {result.total_energy:.1f} kWh")
    lines.append(f"  Completed/to assign: {result.completed_tasks}/{result.unassigned_tasks}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task allocation for inland waterway vessels")
    parser.add_argument("--road-network", type=str,
                        default="/root/demon3.16/src/task_planner/output/road_network.json",
                        help="Path to road network JSON")
    parser.add_argument("--ships", type=str,
                        default="/root/demon3.16/src/task_planner/config/ships.yaml",
                        help="Path to ships YAML config")
    parser.add_argument("--tasks", type=str, default="",
                        help="Path to tasks JSON (if not specified, generates demo tasks)")
    parser.add_argument("--no-2opt", action="store_true",
                        help="Disable 2-opt optimization")
    parser.add_argument("--no-transfer", action="store_true",
                        help="Disable inter-ship transfer optimization")
    parser.add_argument("--output", type=str,
                        default="/root/demon3.16/src/task_planner/output/assignment_result.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Load road network
    from road_network_builder import RoadNetwork
    with open(args.road_network, 'r') as f:
        rn_data = json.load(f)
    network = RoadNetwork.from_dict(rn_data)

    # Build node info map
    node_info = {}
    for node in network.nodes.values():
        if node.is_port:
            node_info[node.id] = node.port_name
        elif node.is_gas_station:
            node_info[node.id] = f"Gas_{node.port_name}"
        else:
            node_info[node.id] = f"N{node.id}"

    # Load ships
    with open(args.ships, 'r') as f:
        ships_config = yaml.safe_load(f)["ships"]

    ships = []
    for sc in ships_config:
        # Find nearest road node
        min_dist = float('inf')
        nearest_node = 0
        for nid, node in network.nodes.items():
            d = np.sqrt((node.x - sc["start_x"])**2 + (node.y - sc["start_y"])**2)
            if d < min_dist:
                min_dist = d
                nearest_node = nid

        ship = Ship(
            id=sc["id"], name=sc["name"],
            max_payload=sc["max_payload"],
            max_energy=sc["max_energy"],
            max_speed=sc["max_speed"],
            position_x=sc["start_x"], position_y=sc["start_y"],
            current_node=nearest_node,
            energy=sc["max_energy"] * 0.9,  # Start at 90% energy
            energy_per_km=sc.get("energy_per_km", 2.5)
        )
        ships.append(ship)

    # Generate demo tasks if not specified
    if args.tasks and os.path.exists(args.tasks):
        with open(args.tasks, 'r') as f:
            tasks_data = json.load(f)
        tasks = [Task.from_dict(td) for td in tasks_data]
    else:
        # Generate demo tasks between ports
        port_nodes = [nid for nid, node in network.nodes.items() if node.is_port]
        print(f"Generating demo tasks using {len(port_nodes)} ports...")
        tasks = []
        task_id = 0
        for i in range(len(port_nodes)):
            for j in range(len(port_nodes)):
                if i != j:
                    pickup = port_nodes[i]
                    delivery = port_nodes[j]
                    d = network.dist_matrix[pickup, delivery]
                    if d < np.inf and d > 0:
                        payload = np.random.choice([300, 500, 800, 1000, 1500])
                        priority = np.random.choice([1, 2, 3])
                        tasks.append(Task(
                            id=task_id,
                            pickup_node=pickup,
                            delivery_node=delivery,
                            payload=payload,
                            start_time=0,
                            deadline=3600 * 8,  # 8 hours
                            priority=priority
                        ))
                        task_id += 1
        print(f"  Generated {len(tasks)} tasks")

    # Run allocation
    print("\nRunning task allocation...")
    result = allocate(
        ships=ships,
        tasks=tasks,
        dist_matrix=network.dist_matrix,
        enable_2opt=not args.no_2opt,
        enable_transfer=not args.no_transfer,
        node_id_to_name=node_info
    )

    # Print result
    output_text = format_allocation_result(result, network.dist_matrix, node_info, "Initial Allocation")
    print(output_text)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nAssignment result saved to: {args.output}")

    # Also save text version
    txt_path = args.output.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write(output_text)
    print(f"Assignment text saved to: {txt_path}")
