#!/usr/bin/env python3
"""
主调度器 — 整合异构图 + Graph Evaluator + ALNS + Route Executor
+ Rolling Horizon + Event Detection + Route Repair

输出: 每艘船的 RoadNode 访问序列 (Route)
"""

import sys, os, numpy as np
from copy import deepcopy
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hetero_graph import build_hetero_graph, update_graph
from graph_evaluator import GraphEvaluator
from alns_scheduler import ALNSScheduler, RouteNode
from route_executor import RouteExecutor
from collections import defaultdict


class EventType:
    ENERGY_LOW = "energy_low"
    FAULT = "fault"
    SHIP_IDLE = "ship_idle"
    NEW_TASK = "new_task"
    TASK_CANCELLED = "task_cancelled"


class EventDetector:
    """状态检测"""
    def __init__(self, energy_threshold_ratio=0.2, health_threshold=0.3):
        self.energy_threshold = energy_threshold_ratio
        self.health_threshold = health_threshold

    def detect(self, ships, tasks, current_time) -> List[dict]:
        events = []
        for sid, ship in ships.items():
            if ship.energy_ratio < self.energy_threshold:
                events.append({'type': EventType.ENERGY_LOW, 'ship_id': sid,
                               'energy': ship.energy_ratio})
            if ship.health < self.health_threshold:
                events.append({'type': EventType.FAULT, 'ship_id': sid,
                               'health': ship.health})
            if ship.is_idle and ship.remaining_capacity > 0:
                pending = [t for t in tasks.values()
                          if t.status in ('pending',) and t.assigned_ship < 0]
                if pending:
                    events.append({'type': EventType.SHIP_IDLE, 'ship_id': sid})
        return events


class RouteRepair:
    """局部 Route 修复"""
    def __init__(self, alns: ALNSScheduler, evaluator: GraphEvaluator):
        self.alns = alns
        self.evaluator = evaluator

    def repair(self, event: dict, ships: Dict, routes: Dict,
                tasks: Dict, frozen_count: int = 2) -> Dict:
        etype = event['type']

        if etype in (EventType.ENERGY_LOW, EventType.FAULT):
            return self._repair_ship_failure(event['ship_id'], ships, routes, tasks, frozen_count)
        elif etype == EventType.SHIP_IDLE:
            return self._repair_idle_ship(event['ship_id'], ships, routes, tasks, frozen_count)
        return routes

    def _repair_ship_failure(self, sid: int, ships: Dict, routes: Dict,
                               tasks: Dict, frozen_count: int) -> Dict:
        """释放故障船的冻结区之后的任务"""
        ship = ships[sid]; route = routes.get(sid, [])
        released = []
        count = 0; new_route = []
        for rn in route:
            count += 1
            if count > frozen_count and rn.task_id >= 0:
                released.append(rn.task_id)
                task = tasks.get(rn.task_id)
                if task: task.assigned_ship = -1
            else:
                new_route.append(rn)
        routes[sid] = new_route

        if released:
            return self.alns.optimize(ships, routes, frozen_count)
        return routes

    def _repair_idle_ship(self, sid: int, ships: Dict, routes: Dict,
                            tasks: Dict, frozen_count: int) -> Dict:
        return self.alns.optimize(ships, routes, frozen_count)


class RollingHorizon:
    """Rolling Horizon 滚动优化"""
    def __init__(self, alns: ALNSScheduler, frozen_count: int = 2):
        self.alns = alns
        self.frozen_count = frozen_count

    def optimize(self, ships: Dict, routes: Dict) -> Dict:
        return self.alns.optimize(ships, routes, self.frozen_count)


class Scheduler:
    """主调度器 — 整合所有模块"""

    def __init__(self, road_network, ships: Dict, tasks: Dict,
                 port_ids: List[int], gas_ids: List[int],
                 node_names: Dict[int, str], alns_config: Dict = None):
        self.rn = road_network
        self.ships = ships
        self.tasks = tasks
        self.node_names = node_names
        self.current_time = 0.0

        self.evaluator = GraphEvaluator(road_network, node_names, tasks)
        self.alns = ALNSScheduler(self.evaluator, tasks, road_network, node_names,
                                  config=alns_config)
        self.executor = RouteExecutor(road_network, tasks, node_names)
        self.detector = EventDetector()
        self.repair = RouteRepair(self.alns, self.evaluator)
        self.rolling = RollingHorizon(self.alns, frozen_count=2)

        self.routes: Dict[int, List[RouteNode]] = {}
        self.event_log: List[dict] = []

    def initialize(self, initial_routes: Dict = None):
        """初始调度: 异构图 + ALNS 生成 Route。
        可传入 initial_routes (如贪心算法结果) 替代 cheapest insertion 初始解。"""
        print("=" * 60)
        print("  Initial Scheduling: Graph + ALNS")
        print("=" * 60)

        g = build_hetero_graph(self.ships, self.tasks, self.rn)
        print(f"  Graph: Ship={g.M} Task={g.K} Road={g.N}")
        print(f"  Edges: RR={g.rr_edges.shape[1]} SR={g.sr_edges.shape[1]} "
              f"TR={g.tr_edges.shape[1]} ST={g.st_edges.shape[1]} TT={g.tt_edges.shape[1]}")

        if initial_routes is not None:
            self.routes = initial_routes
            print("  Using provided initial routes (e.g. greedy result)")
        else:
            self.routes = self.alns.build_initial_routes(self.ships)
        self._print_routes("Initial Routes")

        self.routes = self.alns.optimize(self.ships, self.routes)
        self._print_routes("Optimized Routes (ALNS)")

        return self.routes

    def step(self, dt: float = 300):
        """执行一个时间步"""
        self.current_time += dt

        # 1. 执行 Route
        self.executor.execute_step(self.ships, self.routes, dt, self.current_time)

        # 2. 状态检测
        events = self.detector.detect(self.ships, self.tasks, self.current_time)

        # 3. 处理事件
        for evt in events:
            before = {sid: self.alns.route_to_node_list(self.routes.get(sid, []))
                      for sid in self.ships}
            self.routes = self.repair.repair(evt, self.ships, self.routes, self.tasks)
            after = {sid: self.alns.route_to_node_list(self.routes.get(sid, []))
                     for sid in self.ships}
            if before != after:
                self.event_log.append({
                    'time': self.current_time, 'event': evt,
                    'before': before, 'after': after
                })

        # 4. Rolling Horizon (每完成一个节点触发)
        # 简化: 每 step 检查是否需要优化
        for sid, ship in self.ships.items():
            if ship.current_phase == 'idle' and len(self.routes.get(sid, [])) > 0:
                self.routes = self.rolling.optimize(self.ships, self.routes)

    def run(self, total_time: float, dt: float = 300):
        """运行指定时间"""
        steps = int(total_time / dt)
        for i in range(steps):
            self.step(dt)
            if i % 12 == 0 and i > 0:  # 每小时打印一次
                self._print_status()
        self._print_status()

    def _print_routes(self, title: str):
        print(f"\n  {title}:")
        for sid in sorted(self.ships.keys()):
            ship = self.ships[sid]
            route = self.routes.get(sid, [])
            node_list = self.alns.route_to_node_list(route)
            desc = self.alns.format_route(ship, route)
            print(f"  {ship.name}: {node_list}")
            print(f"           {desc}")

    def _print_status(self):
        print(f"\n  [t={self.current_time/60:.0f}min]")
        for sid, ship in sorted(self.ships.items()):
            route = self.routes.get(sid, [])
            node = self.node_names.get(ship.current_node, f'N{ship.current_node}')
            print(f"  {ship.name} @{node} e={ship.energy:.0f} load={ship.load:.0f}t "
                  f"route={len(route)}nodes done={len(ship.completed_tasks)}")

    def get_node_sequences(self) -> Dict[int, List[int]]:
        """输出纯 RoadNode ID 序列 (给路径规划器)"""
        return {sid: self.alns.route_to_node_list(self.routes.get(sid, []))
                for sid in self.ships}
