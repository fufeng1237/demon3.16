#!/usr/bin/env python3
"""
Task Scheduler Node - 内河船舶任务调度主节点

ROS node integrating:
  - Road network loading
  - Task assignment (greedy + 2-opt)
  - Event monitoring
  - Adaptive reallocation

Publishes:
  /task_planner/road_network_markers  (visualization_msgs/MarkerArray)
  /task_planner/ship_markers          (visualization_msgs/MarkerArray)
  /task_planner/assignment_result     (std_msgs/String - JSON)
  /task_planner/reallocation_log      (std_msgs/String)

Parameters (ROS param server):
  ~csv_path: path to grid map CSV
  ~ports_config: path to ports YAML
  ~gas_stations_config: path to gas stations YAML
  ~ships_config: path to ships YAML
  ~tasks_config: path to tasks JSON (optional, auto-generated if empty)
  ~output_dir: directory for output files
  ~scheduling_interval: seconds between scheduling cycles (default 5.0)
"""

import rospy
import json
import yaml
import os
import sys
import numpy as np
from datetime import datetime
from copy import deepcopy

# Import project modules
from road_network import (
    RoadNetwork, build_road_network, load_grid_csv,
    extract_water_mask, zhang_suen_thinning
)
from task_assigner import (
    Ship, Task, TaskStatus, ShipStatus, Assignment,
    AllocationResult, allocate, format_allocation_result,
    format_ship_decision
)
from event_monitor import (
    EventMonitor, Reallocator, Event, EventType,
    ReallocScope, ReallocationDecision, DeviationVector,
    format_reallocation_output
)
from visualize_road_net import (
    create_road_network_markers, create_ship_markers,
    create_assignment_visualization
)

# ROS message imports
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String, Float32MultiArray


class TaskSchedulerNode:
    """Main ROS node for task scheduling and reallocation."""

    def __init__(self):
        rospy.init_node("task_scheduler_node", anonymous=False)

        # ── Load parameters ──
        csv_path = rospy.get_param("~csv_path",
            "/root/demon3.16/src/waterway_map/map/combined_distance_field.csv")
        ports_config_path = rospy.get_param("~ports_config",
            "/root/demon3.16/src/task_planner/config/ports.yaml")
        gas_config_path = rospy.get_param("~gas_stations_config",
            "/root/demon3.16/src/task_planner/config/gas_stations.yaml")
        ships_config_path = rospy.get_param("~ships_config",
            "/root/demon3.16/src/task_planner/config/ships.yaml")
        tasks_config_path = rospy.get_param("~tasks_config", "")
        output_dir = rospy.get_param("~output_dir",
            "/root/demon3.16/src/task_planner/output")
        self.scheduling_interval = rospy.get_param("~scheduling_interval", 5.0)
        map_resolution = rospy.get_param("~map_resolution", 0.5)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # ── Publishers ──
        self.road_net_pub = rospy.Publisher(
            "~road_network_markers", MarkerArray, queue_size=1, latch=True)
        self.ship_pub = rospy.Publisher(
            "~ship_markers", MarkerArray, queue_size=1, latch=True)
        self.assignment_pub = rospy.Publisher(
            "~assignment_result", String, queue_size=1, latch=True)
        self.reallocation_pub = rospy.Publisher(
            "~reallocation_log", String, queue_size=10)
        self.deviation_pub = rospy.Publisher(
            "~deviations", Float32MultiArray, queue_size=10)

        # ── Step 1: Build/Load Road Network ──
        road_net_path = os.path.join(output_dir, "road_network.json")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Task Scheduler Node Starting...")
        rospy.loginfo("=" * 60)

        if os.path.exists(road_net_path):
            rospy.loginfo(f"[1/4] Loading existing road network: {road_net_path}")
            with open(road_net_path, 'r') as f:
                self.network = RoadNetwork.from_dict(json.load(f))
            rospy.loginfo(f"  Loaded: {len(self.network.nodes)} nodes, "
                          f"{len(self.network.edges)} edges")
        else:
            rospy.loginfo(f"[1/4] Building new road network from CSV: {csv_path}")
            with open(ports_config_path, 'r') as f:
                ports_config = yaml.safe_load(f)["ports"]
            with open(gas_config_path, 'r') as f:
                gas_config = yaml.safe_load(f)["gas_stations"]

            self.network = build_road_network(
                csv_path=csv_path,
                ports_config=ports_config,
                gas_stations_config=gas_config,
                resolution=map_resolution,
                output_dir=output_dir
            )
            # Save for reuse
            with open(road_net_path, 'w') as f:
                json.dump(self.network.to_dict(), f, indent=2)
            rospy.loginfo(f"  Road network saved to: {road_net_path}")

        # Extract helper lists
        self.gas_nodes = [nid for nid, n in self.network.nodes.items() if n.is_gas_station]
        self.port_nodes = [nid for nid, n in self.network.nodes.items() if n.is_port]
        self.node_info = {}
        for nid, node in self.network.nodes.items():
            if node.is_port:
                self.node_info[nid] = node.port_name
            elif node.is_gas_station:
                self.node_info[nid] = f"GS_{node.port_name}"
            else:
                self.node_info[nid] = f"N{nid}"

        # ── Step 2: Load Ships ──
        rospy.loginfo(f"[2/4] Loading ships: {ships_config_path}")
        with open(ships_config_path, 'r') as f:
            ships_config = yaml.safe_load(f)["ships"]
        self.ships = self._create_ships(ships_config)
        for s in self.ships:
            rospy.loginfo(f"  {s.name}: pos=({s.position_x},{s.position_y}), "
                          f"capacity={s.max_payload}t, node={s.current_node}")

        # ── Step 3: Load/Create Tasks ──
        rospy.loginfo(f"[3/4] Loading tasks")
        self.tasks = self._create_tasks(tasks_config_path)
        rospy.loginfo(f"  Created {len(self.tasks)} tasks")

        # ── Step 4: Initial Allocation ──
        rospy.loginfo(f"[4/4] Running initial allocation")
        self.assignments = []
        self.allocation_result = self._run_allocation("Initial Allocation")
        self._publish_allocation_result()

        # ── Initialize monitors ──
        self.event_monitor = EventMonitor(
            energy_threshold=rospy.get_param("~energy_threshold", 0.2),
            health_threshold=rospy.get_param("~health_threshold", 0.3),
        )
        self.reallocator = Reallocator(
            refuel_cost_penalty=rospy.get_param("~refuel_penalty", 0.3),
            transfer_cost_penalty=rospy.get_param("~transfer_penalty", 0.2),
        )

        self.simulation_time = 0.0
        self.reallocation_count = 0
        self.allocation_history = []  # Store (before, after) pairs

        # ── Publish road network markers ──
        self._publish_road_network_markers()

        # ── Timer for periodic scheduling ──
        rospy.Timer(rospy.Duration(self.scheduling_interval), self.scheduling_loop)

        rospy.loginfo("=" * 60)
        rospy.loginfo("Task Scheduler Node initialized. Running...")
        rospy.loginfo(f"  Scheduling interval: {self.scheduling_interval}s")
        rospy.loginfo(f"  Ships: {len(self.ships)}")
        rospy.loginfo(f"  Tasks: {len(self.tasks)}")
        rospy.loginfo(f"  Ports: {len(self.port_nodes)}")
        rospy.loginfo(f"  Gas Stations: {len(self.gas_nodes)}")
        rospy.loginfo("=" * 60)

    # ============================================================
    # Initialization Helpers
    # ============================================================

    def _create_ships(self, ships_config: List[Dict]) -> List[Ship]:
        """Create ship objects and find their nearest road network nodes."""
        ships = []
        for sc in ships_config:
            # Find nearest road node
            min_dist = float('inf')
            nearest_node = 0
            for nid, node in self.network.nodes.items():
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
                position_x=sc["start_x"],
                position_y=sc["start_y"],
                current_node=nearest_node,
                energy=sc["max_energy"] * 0.9,  # Start at 90%
                energy_per_km=sc.get("energy_per_km", 2.5)
            )
            ships.append(ship)
        return ships

    def _create_tasks(self, tasks_config_path: str) -> List[Task]:
        """Create tasks from config or auto-generate."""
        if tasks_config_path and os.path.exists(tasks_config_path):
            with open(tasks_config_path, 'r') as f:
                tasks_data = json.load(f)
            return [Task.from_dict(td) for td in tasks_data]

        # Auto-generate tasks between ports
        rospy.loginfo("  No tasks config provided, auto-generating tasks...")
        tasks = []
        task_id = 0
        for i in range(len(self.port_nodes)):
            for j in range(len(self.port_nodes)):
                if i != j:
                    pickup = self.port_nodes[i]
                    delivery = self.port_nodes[j]
                    d = self.network.dist_matrix[pickup, delivery]
                    if d < np.inf and d > 0:
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
        return tasks

    # ============================================================
    # Allocation
    # ============================================================

    def _run_allocation(self, label: str = "Allocation") -> AllocationResult:
        """Run the full allocation pipeline."""
        return allocate(
            ships=self.ships,
            tasks=self.tasks,
            dist_matrix=self.network.dist_matrix,
            enable_2opt=True,
            enable_transfer=True,
            node_id_to_name=self.node_info
        )

    def _publish_allocation_result(self):
        """Publish allocation result as JSON string."""
        result_json = json.dumps(self.allocation_result.to_dict(), indent=2, ensure_ascii=False)
        self.assignment_pub.publish(String(data=result_json))

        # Also save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(self.output_dir, f"assignment_{timestamp}.json")
        with open(result_path, 'w') as f:
            f.write(result_json)

        # Save text version
        text_output = format_allocation_result(
            self.allocation_result, self.network.dist_matrix, self.node_info,
            "Initial Allocation"
        )
        txt_path = result_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write(text_output)

        rospy.loginfo(f"  Allocation saved: {result_path}")
        rospy.loginfo(text_output)

        # Publish ship markers
        self._publish_ship_markers()

    # ============================================================
    # Scheduling Loop
    # ============================================================

    def scheduling_loop(self, event):
        """Main scheduling loop. Called periodically by ROS timer."""
        self.simulation_time += self.scheduling_interval

        # 1. Update ship states (simulate movement/progress)
        self._simulate_progress()

        # 2. Compute deviation vectors
        global_unassigned = any(
            t.status == TaskStatus.WAITING for t in self.tasks
        )
        deviations = self.event_monitor.compute_deviations(
            self.ships, self.tasks, {},
            global_has_unassigned=global_unassigned
        )

        # Publish deviations
        dev_msg = Float32MultiArray()
        for dv in deviations:
            dev_msg.data.extend(dv.to_list())
        self.deviation_pub.publish(dev_msg)

        # 3. Detect events
        events = self.event_monitor.detect_events(
            self.ships, self.tasks, self.simulation_time
        )

        if not events:
            return

        # 4. Handle events
        for evt in events:
            rospy.logwarn(f"[t={self.simulation_time:.0f}s] Event detected: "
                          f"{evt.type.value} on ship_{evt.ship_id}")

            # Save "before" state
            ships_before = deepcopy(self.ships)
            tasks_before = deepcopy(self.tasks)
            assignments_before = deepcopy(self.assignments)

            # Make reallocation decision
            decision = self.reallocator.decide(
                evt, self.ships, self.tasks, self.network.dist_matrix,
                self.gas_nodes, self.port_nodes, self.assignments
            )

            if decision.triggered:
                rospy.logwarn(f"  → Reallocation triggered: {decision.action}")
                rospy.logwarn(f"  → Scope: {decision.scope.value}")
                rospy.logwarn(f"  → Rationale: {decision.rationale}")

                # Apply reallocation
                self._apply_reallocation(decision)

                # Re-run allocation with updated state
                self.allocation_result = self._run_allocation(
                    f"Reallocation ({evt.type.value})"
                )

                # Save "after" state
                ships_after = deepcopy(self.ships)

                # Format and publish comparison
                realloc_text = format_reallocation_output(
                    decision, ships_before, ships_after, self.tasks
                )
                rospy.loginfo("\n" + realloc_text)
                self.reallocation_pub.publish(String(data=realloc_text))

                # Save reallocation log
                self.reallocation_count += 1
                log_path = os.path.join(
                    self.output_dir,
                    f"reallocation_{self.reallocation_count:04d}.txt"
                )
                with open(log_path, 'w') as f:
                    f.write(realloc_text)

                self.allocation_history.append({
                    "event": decision.event.to_dict(),
                    "before": ships_before,
                    "after": ships_after,
                    "decision": decision.to_dict()
                })

                # Update published state
                self._publish_allocation_result()
            else:
                rospy.loginfo(f"  → No reallocation needed: {decision.rationale}")

    def _simulate_progress(self):
        """Simulate ship progress for demonstration purposes."""
        task_map = {t.id: t for t in self.tasks}
        dt = self.scheduling_interval  # seconds

        for ship in self.ships:
            if not ship.execution_order:
                ship.status = ShipStatus.IDLE
                continue

            # Simulate: each cycle, make progress on current task
            current_tid = ship.execution_order[0]
            current_task = task_map.get(current_tid)

            if current_task and current_task.status == TaskStatus.WAITING:
                current_task.status = TaskStatus.LOADING
                ship.status = ShipStatus.LOADING
            elif current_task and current_task.status == TaskStatus.ASSIGNED:
                current_task.status = TaskStatus.LOADING
                ship.status = ShipStatus.LOADING
            elif current_task and current_task.status == TaskStatus.LOADING:
                # After loading, start transport
                current_task.status = TaskStatus.TRANSPORTING
                ship.status = ShipStatus.MOVING
                ship.load += current_task.payload
            elif current_task and current_task.status == TaskStatus.TRANSPORTING:
                # Simulate travel: reduce distance-based ETA
                remaining = current_task.deadline - self.simulation_time
                if remaining <= 0 or self.simulation_time > current_task.start_time + 1800:
                    # Task completed
                    current_task.status = TaskStatus.FINISHED
                    ship.load -= current_task.payload
                    ship.status = ShipStatus.IDLE

                    # Update ship position to delivery node
                    delivery = self.network.nodes.get(current_task.delivery_node)
                    if delivery:
                        ship.position_x = delivery.x
                        ship.position_y = delivery.y
                        ship.current_node = current_task.delivery_node

                    # Energy consumption
                    d = self.network.dist_matrix.get(current_task.pickup_node,
                                                      current_task.delivery_node)
                    if hasattr(d, '__len__'):  # It's a numpy value
                        pass  # Already scalar
                    ship.energy -= (float(d) / 1000.0) * ship.energy_per_km

            # Update ETA
            if ship.execution_order:
                ship.ETA = self.simulation_time + 600  # 10 min remaining

    def _apply_reallocation(self, decision: ReallocationDecision):
        """Apply a reallocation decision to ship and task state."""
        task_map = {t.id: t for t in self.tasks}
        ship_map = {s.id: s for s in self.ships}

        for tid in decision.released_tasks:
            if tid in task_map:
                task_map[tid].status = TaskStatus.WAITING
                task_map[tid].assigned_ship = -1

        for na in decision.new_assignments:
            sid = na["ship_id"]
            tid = na["task_id"]
            if sid in ship_map and tid in task_map:
                task_map[tid].status = TaskStatus.ASSIGNED
                task_map[tid].assigned_ship = sid
                if tid not in ship_map[sid].task_queue:
                    ship_map[sid].task_queue.append(tid)
                    ship_map[sid].execution_order.append(tid)

    # ============================================================
    # ROS Publishers
    # ============================================================

    def _publish_road_network_markers(self):
        """Publish RViz markers for the road network."""
        markers = create_road_network_markers(self.network, frame_id="map")
        if markers:
            self.road_net_pub.publish(markers)
            rospy.loginfo("  Road network markers published")
            # Also publish assignment visualization
            assign_markers = create_assignment_visualization(
                self.network, self.allocation_result, frame_id="map"
            )
            if assign_markers:
                self.assignment_pub.publish(String(
                    data=json.dumps({"marker_count": len(assign_markers.markers)})
                ))

    def _publish_ship_markers(self):
        """Publish RViz markers for ships."""
        markers = create_ship_markers(self.ships, frame_id="map")
        if markers:
            self.ship_pub.publish(markers)

    # ============================================================
    # Summary Report
    # ============================================================

    def generate_summary_report(self) -> str:
        """Generate a summary report of all scheduling activity."""
        lines = []
        lines.append("=" * 70)
        lines.append("  TASK SCHEDULER SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append(f"  Simulation time: {self.simulation_time:.0f} s")
        lines.append(f"  Ships: {len(self.ships)}")
        lines.append(f"  Tasks: {len(self.tasks)}")
        lines.append(f"  Reallocation events: {self.reallocation_count}")
        lines.append("")
        lines.append("  Ship Status:")
        for ship in self.ships:
            finished = sum(1 for tid in ship.task_queue
                          if self._get_task(tid).status == TaskStatus.FINISHED)
            lines.append(f"    {ship.name}: {finished}/{len(ship.execution_order)} tasks, "
                         f"energy={ship.energy:.0f}kWh, status={ship.status.value}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def _get_task(self, tid: int) -> Task:
        for t in self.tasks:
            if t.id == tid:
                return t
        return None

    def __del__(self):
        """Destructor: save summary report."""
        try:
            report = self.generate_summary_report()
            report_path = os.path.join(self.output_dir, "summary_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            rospy.loginfo(f"\n{report}")
            rospy.loginfo(f"Summary report saved to: {report_path}")
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    try:
        node = TaskSchedulerNode()
        rospy.on_shutdown(lambda: node.generate_summary_report())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Task scheduler node failed: {e}")
        import traceback
        traceback.print_exc()
