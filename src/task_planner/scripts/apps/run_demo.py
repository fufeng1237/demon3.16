#!/usr/bin/env python3
"""
Demo Script - 内河船舶任务分配与自适应重分配完整演示

Runs the full pipeline:
  1. Build/Load road network from PNG map
  2. Generate feasible tasks (between reachable ports)
  3. Run initial task allocation (greedy + 2-opt)
  4. Simulate events and demonstrate reallocation
  5. Output all decisions and logs
"""

import numpy as np
import json
import yaml
import os
import sys
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import (RoadNetwork, build_road_network_from_png)
from task_assigner import (Ship, Task, TaskStatus, ShipStatus, Assignment,
                            AllocationResult, allocate, format_allocation_result,
                            format_ship_decision, compute_sequence_cost,
                            compute_energy_cost, compute_time_cost)
from event_monitor import (EventMonitor, Reallocator, Event, EventType,
                            ReallocScope, ReallocationDecision, DeviationVector,
                            format_reallocation_output)
from visualize_road_net import (visualize_road_network_on_map,
                                  visualize_assignment_on_map,
                                  visualize_road_network_static)


# ============================================================
# Configuration
# ============================================================

@dataclass
class DemoConfig:
    png_path: str = "/root/demon3.16/data/maps/map10.png"
    scaled_binary_path: str = "/root/demon3.16/data/maps/binary_map_scaled.png"
    use_scaled_binary: bool = True
    pixel_scale: float = 2.0  # m/pixel for scaled binary map
    ports_config: str = "/root/demon3.16/src/task_planner/config/ports.yaml"
    gas_stations_config: str = "/root/demon3.16/src/task_planner/config/gas_stations.yaml"
    ships_config: str = "/root/demon3.16/src/task_planner/config/ships.yaml"
    output_dir: str = "/root/demon3.16/src/task_planner/output"
    pixel_scale: float = 0.5         # m/pixel
    downscale: int = 8                # PNG downscale factor
    min_component_size: int = 10      # Min nodes in component to consider
    max_tasks: int = 20               # Max auto-generated tasks


def find_connected_components(rn: RoadNetwork) -> List[Set[int]]:
    """Find all connected components in the road network."""
    adj = defaultdict(list)
    for edge in rn.edges:
        adj[edge.from_id].append(edge.to_id)
        adj[edge.to_id].append(edge.from_id)

    visited = set()
    components = []
    for nid in rn.nodes:
        if nid in visited:
            continue
        stack = [nid]
        comp = set()
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.add(v)
            for nb in adj.get(v, []):
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)

    return sorted(components, key=len, reverse=True)


def generate_feasible_tasks(rn: RoadNetwork,
                             components: List[Set[int]],
                             max_tasks: int = 20) -> List[Task]:
    """
    Generate tasks only between ports in the SAME connected component.
    Ensures all tasks are feasible.
    """
    # Build component lookup
    node_to_comp = {}
    for ci, comp in enumerate(components):
        for nid in comp:
            node_to_comp[nid] = ci

    # Find ports by component
    ports_by_comp = defaultdict(list)
    for nid, node in rn.nodes.items():
        if node.is_port and nid in node_to_comp:
            ports_by_comp[node_to_comp[nid]].append(nid)

    tasks = []
    task_id = 0

    for ci, port_list in ports_by_comp.items():
        if len(port_list) < 2:
            continue  # Need at least 2 ports in same component

        for i, pickup in enumerate(port_list):
            for j, delivery in enumerate(port_list):
                if i == j:
                    continue
                d = rn.dist_matrix[pickup, delivery]
                if d <= 0 or d == np.inf:
                    continue

                payload = np.random.choice([300, 500, 800, 1000, 1500])
                priority = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])

                tasks.append(Task(
                    id=task_id,
                    pickup_node=pickup,
                    delivery_node=delivery,
                    payload=float(payload),
                    start_time=0,
                    deadline=3600 * 8,  # 8 hours
                    priority=priority
                ))
                task_id += 1

                if task_id >= max_tasks:
                    return tasks

    return tasks


def create_ships_from_config(rn: RoadNetwork,
                              ships_config: List[Dict],
                              components: List[Set[int]] = None) -> List[Ship]:
    """
    Create ships and snap to nearest road node in a major component.
    Falls back to nearest node in any component if no major component nearby.
    """
    ships = []
    # Collect nodes from top components (those with ports)
    major_nodes = set()
    if components:
        for comp in components[:8]:  # Top 8 components
            major_nodes.update(comp)

    if not major_nodes:
        major_nodes = set(rn.nodes.keys())

    for sc in ships_config:
        # Find nearest node in major components
        min_dist = float('inf')
        nearest_node = None
        for nid in major_nodes:
            node = rn.nodes[nid]
            d = np.sqrt((node.x - sc["start_x"])**2 +
                       (node.y - sc["start_y"])**2)
            if d < min_dist:
                min_dist = d
                nearest_node = nid

        # Fallback to any node
        if nearest_node is None:
            for nid, node in rn.nodes.items():
                d = np.sqrt((node.x - sc["start_x"])**2 +
                           (node.y - sc["start_y"])**2)
                if d < min_dist:
                    min_dist = d
                    nearest_node = nid

        ship = Ship(
            id=sc["id"],
            name=sc["name"],
            max_payload=sc["max_payload"],
            max_energy=sc["max_energy"],
            max_speed=sc["max_speed"],
            position_x=rn.nodes[nearest_node].x,
            position_y=rn.nodes[nearest_node].y,
            current_node=nearest_node,
            energy=sc["max_energy"] * 0.9,
            energy_per_km=sc.get("energy_per_km", 2.5)
        )
        ships.append(ship)
    return ships


# ============================================================
# Demo Scenarios
# ============================================================

def simulate_energy_low_scenario(ships: List[Ship], ship_id: int):
    """Simulate energy low event on a ship."""
    for s in ships:
        if s.id == ship_id:
            s.energy = s.max_energy * 0.15
            print(f"\n[SCENARIO] {s.name} energy low: {s.energy:.0f}/{s.max_energy:.0f} kWh")
            return

def simulate_fault_scenario(ships: List[Ship], ship_id: int):
    """Simulate fault event on a ship."""
    for s in ships:
        if s.id == ship_id:
            s.health = 0.2
            print(f"\n[SCENARIO] {s.name} fault: health={s.health}")
            return

def simulate_task_delayed_scenario(ships: List[Ship], tasks: List[Task], ship_id: int):
    """Simulate task delay."""
    for s in ships:
        if s.id == ship_id and s.execution_order:
            tid = s.execution_order[0]
            for t in tasks:
                if t.id == tid:
                    t.deadline = 100  # Very tight
                    print(f"\n[SCENARIO] Task {tid} delayed on {s.name}")
                    return


# ============================================================
# Output Formatters
# ============================================================

def build_node_info(rn: RoadNetwork) -> Dict[int, str]:
    """Build node ID → name mapping."""
    info = {}
    for nid, node in rn.nodes.items():
        if node.is_port:
            info[nid] = node.port_name
        elif node.is_gas_station:
            info[nid] = f"GS_{node.port_name}"
        else:
            info[nid] = f"N{nid}"
    return info


def print_road_network_summary(rn: RoadNetwork,
                                components: List[Set[int]]):
    """Print formatted road network summary."""
    ports_by_comp = defaultdict(list)
    gas_by_comp = defaultdict(list)
    node_to_comp = {}
    for ci, comp in enumerate(components):
        for nid in comp:
            node_to_comp[nid] = ci
            node = rn.nodes[nid]
            if node.is_port:
                ports_by_comp[ci].append(node.port_name)
            if node.is_gas_station:
                gas_by_comp[ci].append(node.port_name)

    print("=" * 70)
    print("  ROAD NETWORK STRUCTURE")
    print("=" * 70)
    print(f"  Total nodes: {len(rn.nodes)}")
    print(f"    - Regular: {sum(1 for n in rn.nodes.values() if not n.is_port and not n.is_gas_station)}")
    print(f"    - Ports: {sum(1 for n in rn.nodes.values() if n.is_port)}")
    print(f"    - Gas Stations: {sum(1 for n in rn.nodes.values() if n.is_gas_station)}")
    print(f"  Total edges: {len(rn.edges)}")
    total_len = sum(e.distance for e in rn.edges)
    print(f"  Total waterway length: {total_len:.1f} m ({total_len/1000:.2f} km)")
    print(f"  Connected components: {len(components)}")
    print(f"\n  Major components (>={config.min_component_size} nodes):")
    for ci, comp in enumerate(components):
        if len(comp) >= config.min_component_size:
            ports = ports_by_comp.get(ci, [])
            gs = gas_by_comp.get(ci, [])
            print(f"    C{ci}: {len(comp)} nodes, "
                  f"ports={ports}, gas_stations={gs}")
    print("=" * 70)


# ============================================================
# Main Demo
# ============================================================

if __name__ == "__main__":
    config = DemoConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("  内河船舶任务分配与自适应重分配 - DEMO")
    print("=" * 70)

    # ── Phase 1: Build/Load Road Network ──
    print("\n" + "─" * 70)
    print("  PHASE 1: Road Network Construction")
    print("─" * 70)

    rn_path = os.path.join(config.output_dir, "road_network.json")

    if os.path.exists(rn_path):
        print(f"Loading existing road network: {rn_path}")
        with open(rn_path, 'r') as f:
            rn = RoadNetwork.from_dict(json.load(f))
        print(f"  Loaded: {len(rn.nodes)} nodes, {len(rn.edges)} edges")
    else:
        # Choose map source
        if config.use_scaled_binary:
            map_path = config.scaled_binary_path
            print(f"Building road network from scaled binary map: {map_path}")
        else:
            map_path = config.png_path
            print(f"Building road network from original PNG: {map_path}")

        with open(config.ports_config, 'r') as f:
            ports_cfg = yaml.safe_load(f)["ports"]
        with open(config.gas_stations_config, 'r') as f:
            gas_cfg = yaml.safe_load(f)["gas_stations"]
        rn = build_road_network_from_png(
            png_path=map_path,
            ports_config=ports_cfg,
            gas_stations_config=gas_cfg,
            downscale=config.downscale,
            is_scaled_binary=config.use_scaled_binary,
            pixel_scale=config.pixel_scale,
            output_dir=config.output_dir
        )
        with open(rn_path, 'w') as f:
            json.dump(rn.to_dict(), f, indent=2)

    components = find_connected_components(rn)
    node_info = build_node_info(rn)
    print_road_network_summary(rn, components)

    # ── Phase 2: Generate Tasks ──
    print("\n" + "─" * 70)
    print("  PHASE 2: Task Generation")
    print("─" * 70)

    tasks = generate_feasible_tasks(rn, components, config.max_tasks)
    print(f"Generated {len(tasks)} feasible tasks (between reachable ports)")
    for t in tasks:
        pickup_name = node_info.get(t.pickup_node, f"N{t.pickup_node}")
        delivery_name = node_info.get(t.delivery_node, f"N{t.delivery_node}")
        d = rn.dist_matrix[t.pickup_node, t.delivery_node]
        print(f"  T{t.id}: {pickup_name} → {delivery_name}, "
              f"{t.payload}t, priority={t.priority}, dist={d/1000:.1f}km")

    # ── Phase 3: Load Ships ──
    print("\n" + "─" * 70)
    print("  PHASE 3: Ship Loading")
    print("─" * 70)

    with open(config.ships_config, 'r') as f:
        ships_cfg = yaml.safe_load(f)["ships"]
    ships = create_ships_from_config(rn, ships_cfg, components)
    for s in ships:
        nn = node_info.get(s.current_node, f"N{s.current_node}")
        print(f"  {s.name}: pos=({s.position_x}, {s.position_y}), "
              f"capacity={s.max_payload}t, nearest: {nn}")

    # ── Phase 4: Initial Allocation ──
    print("\n" + "─" * 70)
    print("  PHASE 4: Initial Task Allocation")
    print("─" * 70)

    initial_result = allocate(
        ships=deepcopy(ships),
        tasks=deepcopy(tasks),
        dist_matrix=rn.dist_matrix,
        enable_2opt=True,
        enable_transfer=True,
        node_id_to_name=node_info
    )

    initial_text = format_allocation_result(
        initial_result, rn.dist_matrix, node_info, "INITIAL ALLOCATION"
    )
    print(initial_text)

    # Save initial allocation
    alloc_path = os.path.join(config.output_dir, "initial_allocation.txt")
    with open(alloc_path, 'w') as f:
        f.write(initial_text)
    alloc_json_path = os.path.join(config.output_dir, "initial_allocation.json")
    try:
        with open(alloc_json_path, 'w') as f:
            json.dump(initial_result.to_dict(), f, indent=2)
    except (TypeError, ValueError) as e:
        print(f"  WARNING: Could not save JSON: {e}")

    # ── Phase 5: Reallocation Scenarios ──
    print("\n" + "─" * 70)
    print("  PHASE 5: Reallocation Scenarios")
    print("─" * 70)

    gas_nodes = [nid for nid, n in rn.nodes.items() if n.is_gas_station]
    port_nodes = [nid for nid, n in rn.nodes.items() if n.is_port]

    monitor = EventMonitor()
    reallocator = Reallocator()

    all_logs = []

    # Scenario 1: Energy Low
    scenario_ships = deepcopy(initial_result.ships)
    scenario_tasks = deepcopy(initial_result.tasks)
    simulate_energy_low_scenario(scenario_ships, ship_id=0)

    events = monitor.detect_events(scenario_ships, scenario_tasks)
    for evt in events:
        ships_before = deepcopy(scenario_ships)
        decision = reallocator.decide(
            evt, scenario_ships, scenario_tasks, rn.dist_matrix,
            gas_nodes, port_nodes, []
        )

        # Re-run allocation after reallocation if triggered
        if decision.triggered:
            realloc_result = allocate(
                ships=scenario_ships,
                tasks=scenario_tasks,
                dist_matrix=rn.dist_matrix,
                enable_2opt=True,
                enable_transfer=True,
                node_id_to_name=node_info
            )
            realloc_text = format_reallocation_output(
                decision, ships_before, realloc_result.ships, scenario_tasks
            )
            print(realloc_text)

            all_logs.append({
                "type": "reallocation",
                "event": decision.event.to_dict(),
                "decision": decision.to_dict(),
                "before_ships": [s.to_dict() for s in ships_before],
                "after_ships": [s.to_dict() for s in realloc_result.ships]
            })

            # Update state
            scenario_ships = realloc_result.ships
            scenario_tasks = realloc_result.tasks

            # Also show final allocation
            final_text = format_allocation_result(
                realloc_result, rn.dist_matrix, node_info,
                "ALLOCATION AFTER REALLOCATION (Energy Low)"
            )
            print(final_text)
        else:
            print(f"\n  Event: {evt.type.value} on Ship_{evt.ship_id}")
            print(f"  Decision: NOT triggered — {decision.rationale}")

    # Scenario 2: Task Delayed
    scenario_ships2 = deepcopy(scenario_ships)
    scenario_tasks2 = deepcopy(scenario_tasks)
    simulate_task_delayed_scenario(scenario_ships2, scenario_tasks2, ship_id=1)

    events2 = monitor.detect_events(scenario_ships2, scenario_tasks2)
    for evt in events2:
        ships_before2 = deepcopy(scenario_ships2)
        decision2 = reallocator.decide(
            evt, scenario_ships2, scenario_tasks2, rn.dist_matrix,
            gas_nodes, port_nodes, []
        )

        if decision2.triggered:
            realloc_result2 = allocate(
                ships=scenario_ships2,
                tasks=scenario_tasks2,
                dist_matrix=rn.dist_matrix,
                enable_2opt=True, enable_transfer=True,
                node_id_to_name=node_info
            )
            realloc_text2 = format_reallocation_output(
                decision2, ships_before2, realloc_result2.ships, scenario_tasks2
            )
            print(realloc_text2)

            all_logs.append({
                "type": "reallocation",
                "event": decision2.event.to_dict(),
                "decision": decision2.to_dict(),
                "before_ships": [s.to_dict() for s in ships_before2],
                "after_ships": [s.to_dict() for s in realloc_result2.ships]
            })

            final_text2 = format_allocation_result(
                realloc_result2, rn.dist_matrix, node_info,
                "ALLOCATION AFTER REALLOCATION (Task Delayed)"
            )
            print(final_text2)
        else:
            print(f"\n  Event: {evt.type.value} on Ship_{evt.ship_id}")
            print(f"  Decision: NOT triggered — {decision2.rationale}")

    # ── Save All Logs ──
    log_path = os.path.join(config.output_dir, "demo_logs.json")
    with open(log_path, 'w') as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print(f"  Output files:")
    print(f"    Road network:     {rn_path}")
    print(f"    Initial alloc:    {alloc_path}")
    print(f"    Demo logs:        {log_path}")
    print("=" * 70)

    # ── Phase 6: Visualization (overlay on original map) ──
    print("\n" + "─" * 70)
    print("  PHASE 6: Visualization")
    print("─" * 70)
    try:
        # 6a: Road network overlaid on original map
        viz_path = os.path.join(config.output_dir, "road_network_on_map.png")
        visualize_road_network_on_map(
            rn, config.png_path, viz_path,
            display_downscale=config.downscale
        )
        print(f"  [1/2] Road network on map: {viz_path}")

        # 6b: Task assignments overlaid on original map
        assign_viz_path = os.path.join(config.output_dir, "task_assignments_on_map.png")
        visualize_assignment_on_map(
            rn, initial_result, config.png_path, assign_viz_path,
            display_downscale=config.downscale
        )
        print(f"  [2/2] Task assignments on map: {assign_viz_path}")
    except Exception as e:
        print(f"  Visualization error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to graph-only
        try:
            viz_path2 = os.path.join(config.output_dir, "road_network.png")
            visualize_road_network_static(rn, viz_path2)
            print(f"  Fallback visualization: {viz_path2}")
        except Exception:
            pass

    print("\nDone!")
