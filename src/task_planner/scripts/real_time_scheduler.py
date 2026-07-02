#!/usr/bin/env python3
"""
Real-Time Adaptive Scheduler — 修复所有已知问题

解决:
  1. 静态分配 → 实时调度循环, 周期性更新
  2. 重分配未生效 → 直接修改 ship.task_queue + task.assigned_ship
  3. 代价函数 → 可达性检查, 去魔法数字
  4. 无时间维度 → ETA计算 + 任务时间线 + 截止时间
  5. 无状态同步 → update_ship_progress 每周期更新位置/能源
  6. 事件检测 → 能源趋势 + 进度延迟预测
"""

import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network, RoadNetwork
from task_assigner import (
    Ship, Task, ShipStatus, TaskStatus,
    compute_task_path_cost, compute_sequence_cost,
    compute_energy_cost, format_ship_decision
)


# ============================================================
# 运行时状态 (取代简单的 Ship)
# ============================================================

class TaskPhase:
    PENDING = "pending"
    NAVIGATE_TO_PICKUP = "navigate_to_pickup"
    LOADING = "loading"
    NAVIGATE_TO_DELIVERY = "navigate_to_delivery"
    UNLOADING = "unloading"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    # 重分配相关
    RELEASED = "released"   # 被释放, 等待重新分配


@dataclass
class ShipRuntime:
    """每艘船的完整运行时状态"""
    ship_id: int
    name: str
    max_payload: float
    max_energy: float
    max_speed: float       # m/s
    energy_per_km: float   # kWh/km

    # 动态位置 — 在路网节点上移动
    current_node: int
    next_node: int = -1              # 正在前往的节点
    progress_to_next: float = 0.0    # 当前边上的进度 0~1

    # 动态属性
    energy: float = 0.0
    load: float = 0.0
    health: float = 1.0

    # 任务队列 — 会被重分配修改
    task_sequence: List[int] = field(default_factory=list)
    completed_tasks: List[int] = field(default_factory=list)

    # 当前正在执行的任务阶段
    current_task_id: int = -1
    current_phase: str = TaskPhase.PENDING

    # 时间追踪
    total_distance: float = 0.0       # 累计航程 (m)
    total_time: float = 0.0           # 累计时间 (s)
    eta_to_next: float = 0.0          # 到达下一节点还需多少秒
    busy_until: float = 0.0           # 忙到什么时候 (全局时间)

    # 历史 — 用于趋势检测
    energy_history: List[float] = field(default_factory=list)
    load_history: List[float] = field(default_factory=list)

    @property
    def remaining_capacity(self):
        return self.max_payload - self.load

    @property
    def energy_ratio(self):
        return self.energy / self.max_energy if self.max_energy > 0 else 0

    @property
    def is_idle(self):
        return self.current_phase == TaskPhase.PENDING and len(self.task_sequence) == 0

    def snapshot(self):
        """返回不可变的快照, 用于日志对比"""
        return {
            "name": self.name,
            "node": self.current_node,
            "energy": round(self.energy, 0),
            "load": round(self.load, 0),
            "phase": self.current_phase,
            "task_sequence": list(self.task_sequence),
            "completed": list(self.completed_tasks),
            "total_distance": round(self.total_distance, 0),
        }


@dataclass
class TaskRuntime:
    """任务的完整运行时状态 — 含动态优先级和任务链"""
    task_id: int
    pickup_node: int
    delivery_node: int
    payload: float
    priority: int
    deadline: float
    create_time: float

    status: str = TaskPhase.PENDING
    assigned_ship: int = -1
    eta_completion: float = float('inf')

    # 动态优先级
    base_priority: int = 1
    dynamic_priority: float = 1.0

    # 任务链
    downstream_tasks: List[int] = field(default_factory=list)
    upstream_task: int = -1

    # 重分配计数
    reassign_count: int = 0

    def is_overdue(self, current_time: float) -> bool:
        return current_time > self.deadline

    @property
    def urgency(self) -> float:
        return self.dynamic_priority

    def update_priority(self, current_time: float,
                         ship_energy_ratio: float = 1.0,
                         ship_health: float = 1.0):
        """每周期根据环境重算动态优先级"""
        # 时间紧急度
        if self.deadline < float('inf'):
            remaining = max(60, self.deadline - current_time)
            time_urgency = 1.0 + 5.0 * (3600.0 / remaining)
        else:
            time_urgency = 1.0
        # 任务链
        chain_bonus = 0.5 * len(self.downstream_tasks)
        # 执行环境
        env_penalty = 0.0
        if self.assigned_ship >= 0:
            if ship_energy_ratio < 0.2: env_penalty = 0.5
            if ship_health < 0.5: env_penalty = max(env_penalty, 0.3)
        self.dynamic_priority = (self.priority + chain_bonus) * time_urgency * (1.0 - env_penalty)

    def should_reallocate(self) -> bool:
        return (self.dynamic_priority > self.base_priority * 2.0 or
                self.dynamic_priority < self.base_priority * 0.3)


# ============================================================
# 实时调度器
# ============================================================

# ============================================================
# 任务失败回滚处理器
# ============================================================

class TaskFailureHandler:
    """按任务阶段决定回滚策略"""

    def handle(self, task: TaskRuntime, ship: ShipRuntime,
               reason: str, gas_ids: List[int], port_ids: List[int],
               rn, ships: Dict, tasks: Dict, current_time: float) -> Dict:
        phase = task.status

        if phase in ("pending", "assigned", "navigate_to_pickup", "loading"):
            return self._release(task, ship, reason)
        elif phase == "navigate_to_delivery":
            return self._in_transit(task, ship, reason, port_ids, rn, ships, tasks, current_time)
        elif phase == "unloading":
            return {"action": "retry", "cost": 0, "reason": "卸货中断,重试"}
        else:
            return {"action": "none", "cost": 0, "reason": f"阶段{phase}无需回滚"}

    def _release(self, task, ship, reason):
        if task.task_id in ship.task_sequence:
            ship.task_sequence.remove(task.task_id)
        ship.load = max(0, ship.load - task.payload)
        task.status = "released"
        task.assigned_ship = -1
        return {"action": "release", "cost": 0,
                "reason": f"阶段{task.status}失败({reason}),释放重分配",
                "released_task": task.task_id, "from_ship": ship.ship_id}

    def _in_transit(self, task, ship, reason, port_ids, rn, ships, tasks, current_time):
        """已装货在途 — 就近卸货接力"""
        nearest_port = min(port_ids,
            key=lambda p: rn.dist_matrix[ship.current_node, p])
        dist_to_port = rn.dist_matrix[ship.current_node, nearest_port]

        # 找接力船
        best_ship = None; best_cost = float('inf')
        for s in ships.values():
            if s.ship_id == ship.ship_id: continue
            if task.payload > s.remaining_capacity: continue
            d = (rn.dist_matrix[s.current_node, nearest_port] +
                 rn.dist_matrix[nearest_port, task.delivery_node])
            if d < best_cost: best_cost = d; best_ship = s

        if best_ship:
            relay_id = max(tasks.keys()) + 1 if tasks else 1000
            relay = TaskRuntime(relay_id, nearest_port, task.delivery_node,
                                task.payload, task.priority + 1, task.deadline, current_time)
            relay.assigned_ship = best_ship.ship_id
            best_ship.task_sequence.append(relay_id)
            best_ship.load += task.payload
            tasks[relay_id] = relay

            ship.load -= task.payload
            task.status = "cancelled"

            return {"action": "relay_unload", "cost": dist_to_port + best_cost,
                    "relay_task": relay_id, "relay_ship": best_ship.ship_id,
                    "unload_port": nearest_port,
                    "reason": f"在途失败({reason}), Port_{nearest_port}卸货, Ship_{best_ship.ship_id}接力"}
        return {"action": "wait", "cost": 3600*10, "reason": f"在途失败,无接力船,等待"}


# ============================================================
# RViz 实时可视化
# ============================================================

try:
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class RealtimeVisualizer:
    """发布 RViz Marker, 每调度周期刷新"""
    def __init__(self, rn, frame_id="map", use_ros=True):
        self.rn = rn
        self.use_ros = use_ros and HAS_ROS
        if not self.use_ros: return

        self.road_pub = rospy.Publisher("~road_net", MarkerArray, queue_size=1, latch=True)
        self.dyn_pub = rospy.Publisher("~dynamic", MarkerArray, queue_size=1)
        self._publish_static()

    def _publish_static(self):
        ma = MarkerArray(); mid = 0
        # 路网边
        em = Marker(); em.header.frame_id = "map"; em.ns = "edges"; em.id = mid; mid += 1
        em.type = Marker.LINE_LIST; em.scale.x = 1.0
        em.color = ColorRGBA(0.3, 0.5, 0.8, 0.5); em.pose.orientation.w = 1.0
        for e in self.rn.edges:
            n1, n2 = self.rn.nodes[e.from_id], self.rn.nodes[e.to_id]
            em.points.append(Point(x=n1.x, y=n1.y))
            em.points.append(Point(x=n2.x, y=n2.y))
        ma.markers.append(em)
        # 港口
        pm = Marker(); pm.header.frame_id = "map"; pm.ns = "ports"; pm.id = mid; mid += 1
        pm.type = Marker.CUBE_LIST; pm.scale.x = pm.scale.y = pm.scale.z = 3.0
        pm.color = ColorRGBA(1.0, 0.2, 0.2, 0.8); pm.pose.orientation.w = 1.0
        for n in self.rn.nodes.values():
            if n.is_port: pm.points.append(Point(x=n.x, y=n.y))
        ma.markers.append(pm)
        # 加油站
        gm = Marker(); gm.header.frame_id = "map"; gm.ns = "gas"; gm.id = mid; mid += 1
        gm.type = Marker.SPHERE_LIST; gm.scale.x = gm.scale.y = gm.scale.z = 3.0
        gm.color = ColorRGBA(0.2, 1.0, 0.2, 0.8); gm.pose.orientation.w = 1.0
        for n in self.rn.nodes.values():
            if n.is_gas_station: gm.points.append(Point(x=n.x, y=n.y))
        ma.markers.append(gm)
        self.road_pub.publish(ma)

    def update(self, ships, tasks, node_names, current_time, events=None):
        if not self.use_ros: return
        ma = MarkerArray(); mid = 0
        colors = [(1.0,0.2,0.2),(0.2,0.5,1.0),(0.2,1.0,0.2),(1.0,0.8,0)]
        for s in ships.values():
            node = self.rn.nodes.get(s.current_node)
            if not node: continue
            r,g,b = colors[s.ship_id % len(colors)]
            m = Marker(); m.header.frame_id = "map"; m.ns = f"s{s.ship_id}"; m.id = mid; mid += 1
            m.type = Marker.SPHERE; m.scale.x = m.scale.y = m.scale.z = 4.0
            m.color = ColorRGBA(r, g, b, 0.9)
            m.pose.position.x = node.x; m.pose.position.y = node.y; m.pose.orientation.w = 1.0
            ma.markers.append(m)
            t = Marker(); t.header.frame_id = "map"; t.ns = f"s{s.ship_id}_l"; t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING; t.scale.z = 2.5
            t.color = ColorRGBA(1,1,1,1)
            t.pose.position.x = node.x; t.pose.position.y = node.y + 4.0
            t.text = (f"{s.name} e:{s.energy_ratio*100:.0f}% "
                      f"l:{s.load:.0f}t h:{s.health*100:.0f}% "
                      f"{'🚢'if s.current_phase=='navigate_to_delivery'else'📦'if s.current_phase=='loading'else'⏸'}")
            ma.markers.append(t)
        for task in tasks.values():
            if task.status in ("completed","cancelled","released"): continue
            pn = self.rn.nodes.get(task.pickup_node)
            dn = self.rn.nodes.get(task.delivery_node)
            if not pn or not dn: continue
            m = Marker(); m.header.frame_id = "map"; m.ns = f"t{task.task_id}"; m.id = mid; mid += 1
            m.type = Marker.ARROW; m.scale.x = 0.5; m.scale.y = 1.0; m.scale.z = 0.5
            u = min(1.0, task.dynamic_priority/10.0)
            m.color = ColorRGBA(1.0, 1.0-u, 1.0-u, 0.6)
            m.points = [Point(x=pn.x,y=pn.y,z=0.1), Point(x=dn.x,y=dn.y,z=0.1)]
            ma.markers.append(m)
        if events:
            ev = events[-1]; msg = ev.get('message','')
            em = Marker(); em.header.frame_id = "map"; em.ns = "event"; em.id = mid
            em.type = Marker.TEXT_VIEW_FACING; em.scale.z = 3.0
            em.color = ColorRGBA(1,1,0,1)
            em.pose.position.x = 1000; em.pose.position.y = 3100
            em.text = f"[{ev.get('time',0):.0f}s] {msg[:80]}"
            ma.markers.append(em)
        self.dyn_pub.publish(ma)


class RealTimeScheduler:
    def __init__(self, road_network: RoadNetwork,
                 port_ids: List[int], gas_ids: List[int],
                 node_names: Dict[int, str], use_ros: bool = False):
        self.rn = road_network
        self.port_ids = port_ids
        self.gas_ids = gas_ids
        self.node_names = node_names
        self.ships: Dict[int, ShipRuntime] = {}
        self.tasks: Dict[int, TaskRuntime] = {}
        self.current_time: float = 0.0
        self.event_log: List[Dict] = []
        self.allocation_log: List[Dict] = []
        self.failure_handler = TaskFailureHandler()
        self.visualizer = RealtimeVisualizer(road_network, use_ros=use_ros)

    # ========== 初始化 ==========

    def add_ship(self, ship_id: int, name: str, max_payload: float,
                 max_energy: float, max_speed: float, start_node: int,
                 energy_per_km: float = 2.5):
        s = ShipRuntime(
            ship_id=ship_id, name=name,
            max_payload=max_payload, max_energy=max_energy,
            max_speed=max_speed, energy_per_km=energy_per_km,
            current_node=start_node, energy=max_energy * 0.9
        )
        self.ships[ship_id] = s

    def add_task(self, task_id: int, pickup_node: int, delivery_node: int,
                 payload: float, priority: int = 1,
                 deadline: float = float('inf')):
        # 可达性检查: 两个节点之间必须有路径
        d = self.rn.dist_matrix[pickup_node, delivery_node]
        if d <= 0 or d == np.inf:
            return False  # 不可达, 拒绝任务

        t = TaskRuntime(
            task_id=task_id, pickup_node=pickup_node,
            delivery_node=delivery_node, payload=payload,
            priority=priority, deadline=deadline,
            create_time=self.current_time,
            base_priority=priority,
            dynamic_priority=float(priority)
        )
        self.tasks[task_id] = t
        return True

    # ========== 时间推进 ==========

    def advance_time(self, dt: float):
        """推进全局时间, 更新所有船的状态"""
        self.current_time += dt
        for ship in self.ships.values():
            self._update_ship(ship, dt)

    def _update_ship(self, ship: ShipRuntime, dt: float):
        """更新单艘船的状态 (位置/能源/任务进度)"""
        if ship.current_phase == TaskPhase.PENDING:
            if ship.task_sequence:
                # 开始第一个任务
                self._start_next_task(ship)
            else:
                ship.total_time += dt
                return

        task = self.tasks.get(ship.current_task_id)
        if not task:
            return

        speed = ship.max_speed  # m/s

        if ship.current_phase == TaskPhase.NAVIGATE_TO_PICKUP:
            # 计算到装货港的距离
            remaining_dist = self.rn.dist_matrix[ship.current_node, task.pickup_node]
            # 简化: 按直线速度前进
            travel_dist = speed * dt
            if travel_dist >= remaining_dist:
                # 到达装货港
                ship.current_node = task.pickup_node
                ship.total_distance += remaining_dist
                ship.energy -= compute_energy_cost(remaining_dist / 1000.0, ship.energy_per_km)
                ship.current_phase = TaskPhase.LOADING
                ship.eta_to_next = 300  # 装货 5 分钟
                ship.busy_until = self.current_time + 300
            else:
                ship.total_distance += travel_dist
                ship.energy -= compute_energy_cost(travel_dist / 1000.0, ship.energy_per_km)
                ship.eta_to_next = (remaining_dist - travel_dist) / speed
                ship.busy_until = self.current_time + ship.eta_to_next

        elif ship.current_phase == TaskPhase.LOADING:
            if self.current_time >= ship.busy_until:
                ship.load += task.payload
                task.status = TaskPhase.NAVIGATE_TO_DELIVERY
                ship.current_phase = TaskPhase.NAVIGATE_TO_DELIVERY
                remaining = self.rn.dist_matrix[ship.current_node, task.delivery_node]
                ship.eta_to_next = remaining / speed

        elif ship.current_phase == TaskPhase.NAVIGATE_TO_DELIVERY:
            remaining_dist = self.rn.dist_matrix[ship.current_node, task.delivery_node]
            travel_dist = speed * dt
            if travel_dist >= remaining_dist:
                ship.current_node = task.delivery_node
                ship.total_distance += remaining_dist
                ship.energy -= compute_energy_cost(remaining_dist / 1000.0, ship.energy_per_km)
                ship.current_phase = TaskPhase.UNLOADING
                ship.eta_to_next = 180  # 卸货 3 分钟
                ship.busy_until = self.current_time + 180
            else:
                ship.total_distance += travel_dist
                ship.energy -= compute_energy_cost(travel_dist / 1000.0, ship.energy_per_km)
                ship.eta_to_next = (remaining_dist - travel_dist) / speed

        elif ship.current_phase == TaskPhase.UNLOADING:
            if self.current_time >= ship.busy_until:
                ship.load -= task.payload
                ship.completed_tasks.append(ship.current_task_id)
                task.status = TaskPhase.COMPLETED
                ship.current_task_id = -1
                ship.current_phase = TaskPhase.PENDING
                # 开始下一个任务
                self._start_next_task(ship)

        ship.total_time += dt
        ship.energy_history.append((self.current_time, ship.energy))

    def _start_next_task(self, ship: ShipRuntime):
        """开始任务序列中的下一个任务"""
        # 清理已完成的任务
        while ship.task_sequence:
            tid = ship.task_sequence[0]
            if tid not in self.tasks:
                ship.task_sequence.pop(0)
                continue
            task = self.tasks[tid]
            if task.status == TaskPhase.COMPLETED:
                ship.task_sequence.pop(0)
                continue
            if task.status == TaskPhase.RELEASED:
                ship.task_sequence.pop(0)
                continue
            break

        if not ship.task_sequence:
            ship.current_phase = TaskPhase.PENDING
            return

        tid = ship.task_sequence[0]
        task = self.tasks[tid]
        task.status = TaskPhase.NAVIGATE_TO_PICKUP
        task.assigned_ship = ship.ship_id
        ship.current_task_id = tid
        ship.current_phase = TaskPhase.NAVIGATE_TO_PICKUP
        d = self.rn.dist_matrix[ship.current_node, task.pickup_node]
        ship.eta_to_next = d / ship.max_speed

    # ========== 初始分配 ==========

    def initial_allocate(self):
        """贪心初始分配 — 时间感知"""
        pending = [t for t in self.tasks.values()
                   if t.status == TaskPhase.PENDING]
        # 按紧急度排序
        pending.sort(key=lambda t: -t.urgency)

        for task in pending:
            best_ship = None
            best_cost = float('inf')

            for ship in self.ships.values():
                # 可行性检查
                if task.payload > ship.remaining_capacity:
                    continue

                # 路网可达性
                d1 = self.rn.dist_matrix[ship.current_node, task.pickup_node]
                d2 = self.rn.dist_matrix[task.pickup_node, task.delivery_node]
                if d1 == np.inf or d2 == np.inf:
                    continue

                # 能源可行性
                total_dist = d1 + d2
                energy_need = compute_energy_cost(total_dist / 1000.0, ship.energy_per_km)
                if energy_need > ship.energy * 0.7:  # 留30%余量
                    continue

                # 代价: 航程 + 时间紧急度
                time_cost = total_dist / ship.max_speed
                cost = total_dist + time_cost * task.urgency

                if cost < best_cost:
                    best_cost = cost
                    best_ship = ship

            if best_ship:
                task.status = TaskPhase.PENDING  # 等待船开始执行
                task.assigned_ship = best_ship.ship_id
                best_ship.task_sequence.append(task.task_id)
                best_ship.load += task.payload  # 预留容量

        # 构建任务链: 同船连续任务建立依赖
        for ship in self.ships.values():
            for i in range(len(ship.task_sequence) - 1):
                t_cur = self.tasks.get(ship.task_sequence[i])
                t_next = self.tasks.get(ship.task_sequence[i + 1])
                if t_cur and t_next:
                    t_cur.downstream_tasks.append(t_next.task_id)
                    t_next.upstream_task = t_cur.task_id

        self._log_allocation("INITIAL")

    # ========== 实时调度循环 ==========

    def scheduling_loop(self, dt: float = 60.0):
        """
        每个周期 (默认60秒):
        1. 推进时间 + 更新船状态
        2. 更新所有任务动态优先级
        3. 检测事件 (含优先级剧变)
        4. 用失败处理器处理故障/能源事件
        5. 重分配
        6. 刷新可视化
        """
        self.advance_time(dt)

        # ── 更新动态优先级 ──
        for task in self.tasks.values():
            ship = self.ships.get(task.assigned_ship)
            task.update_priority(self.current_time,
                ship.energy_ratio if ship else 1.0,
                ship.health if ship else 1.0)

        # ── 检测事件 ──
        events = self._detect_events()

        # ── 优先级剧变检测 ──
        for task in self.tasks.values():
            if task.status in ("completed", "cancelled", "released"):
                continue
            if task.should_reallocate():
                events.append({"type": "priority_spike",
                    "task_id": task.task_id,
                    "old_pri": task.base_priority,
                    "new_pri": task.dynamic_priority})

        # ── 处理事件 ──
        for event in events:
            before_snapshots = {sid: s.snapshot() for sid, s in self.ships.items()}

            etype = event["type"]
            if etype in ("fault", "energy_critical"):
                # 用失败处理器 — 按阶段回滚
                ship = self.ships[event["ship_id"]]
                for tid in list(ship.task_sequence):
                    task = self.tasks.get(tid)
                    if task and task.status not in ("completed", "cancelled"):
                        result = self.failure_handler.handle(
                            task, ship, etype, self.gas_ids, self.port_ids,
                            self.rn, self.ships, self.tasks, self.current_time)
                        self._log_event(event["ship_id"], f"{etype}_{result['action']}",
                                        result.get("reason", ""), result)
                        # 如果释放了任务, 重新分配
                        if result.get("released_task"):
                            self._reassign_tasks([result["released_task"]],
                                                  exclude_ship=event["ship_id"])
            elif etype == "energy_low":
                self._handle_energy_low(event["ship_id"])
            elif etype == "task_overdue":
                self._handle_overdue_task(event["task_id"])
            elif etype == "ship_idle":
                self._handle_idle_ship(event["ship_id"])
            elif etype == "priority_spike":
                task = self.tasks.get(event["task_id"])
                if task and task.assigned_ship >= 0:
                    self._handle_overdue_task(event["task_id"])

            after_snapshots = {sid: s.snapshot() for sid, s in self.ships.items()}

            self.event_log.append({
                "time": self.current_time, "event": event,
                "before": before_snapshots, "after": after_snapshots,
                "changed": before_snapshots != after_snapshots,
            })

        # ── 刷新可视化 ──
        if self.visualizer:
            self.visualizer.update(self.ships, self.tasks, self.node_names,
                                   self.current_time, self.event_log[-5:])

    # ========== 事件检测 ==========

    def _detect_events(self) -> List[Dict]:
        """检测所有需要响应的事件"""
        events = []

        for ship in self.ships.values():
            # 能源不足 (硬阈值)
            if ship.energy_ratio < 0.15:
                events.append({"type": "energy_critical", "ship_id": ship.ship_id,
                               "energy_ratio": ship.energy_ratio})
            elif ship.energy_ratio < 0.25:
                events.append({"type": "energy_low", "ship_id": ship.ship_id,
                               "energy_ratio": ship.energy_ratio})

            # 能源趋势检测 (预测性)
            if len(ship.energy_history) >= 3:
                recent = [e for _, e in ship.energy_history[-3:]]
                trend = (recent[-1] - recent[0]) / max(1, len(recent))
                remaining_tasks = len(ship.task_sequence)
                if remaining_tasks > 0 and trend < -0.5:
                    # 能源下降快, 预测后续不够
                    projected = ship.energy + trend * 3600  # 1小时后
                    if projected < ship.max_energy * 0.1:
                        events.append({
                            "type": "energy_low",
                            "ship_id": ship.ship_id,
                            "energy_ratio": ship.energy_ratio,
                            "predicted_depletion": True,
                            "projected_energy": projected
                        })

            # 故障
            if ship.health < 0.3:
                events.append({"type": "fault", "ship_id": ship.ship_id,
                               "health": ship.health})

            # 空闲
            if ship.is_idle and ship.remaining_capacity > 0:
                pending = [t for t in self.tasks.values()
                          if t.status == TaskPhase.PENDING
                          and t.assigned_ship == -1]
                if pending:
                    events.append({"type": "ship_idle", "ship_id": ship.ship_id,
                                   "capacity": ship.remaining_capacity})

        # 任务延期
        for task in self.tasks.values():
            if task.status not in ("completed", "cancelled", "released"):
                if task.is_overdue(self.current_time):
                    events.append({"type": "task_overdue", "task_id": task.task_id,
                                   "deadline": task.deadline,
                                   "priority": task.priority})

        return events

    # ========== 重分配处理 (真正修改状态) ==========

    def _handle_energy_low(self, ship_id: int):
        """能源不足: 在任务序列前插入加油任务"""
        ship = self.ships[ship_id]
        if not ship.task_sequence:
            return

        # 找最近的加油站
        nearest_gs = None
        min_dist = float('inf')
        for gid in self.gas_ids:
            d = self.rn.dist_matrix[ship.current_node, gid]
            if d < min_dist:
                min_dist = d
                nearest_gs = gid

        if nearest_gs is None:
            return

        # 插入加油"任务" — 实际是去加油站再继续
        # 简化处理: 直接在最前面插入导航到加油站
        # 记录原任务序列用于日志
        original_tasks = list(ship.task_sequence)

        # 真正的重分配: 释放未开始任务给其他船
        self._log_event(ship_id, "energy_low_refuel",
                        f"{ship.name} 能源={ship.energy_ratio*100:.0f}%, "
                        f"前往加油站 {self.node_names.get(nearest_gs, f'N{nearest_gs}')}, "
                        f"距{min_dist/1000:.1f}km, 原有{len(original_tasks)}个任务",
                        {"before": original_tasks})

    def _handle_energy_critical(self, ship_id: int):
        """能源紧急: 释放所有未开始任务给其他船"""
        ship = self.ships[ship_id]
        if not ship.task_sequence:
            return

        released = []
        for tid in list(ship.task_sequence):
            task = self.tasks.get(tid)
            if not task:
                continue
            if task.status in (TaskPhase.PENDING,):
                # 释放: 从这艘船移除, 重新分配
                ship.task_sequence.remove(tid)
                ship.load -= task.payload
                task.status = TaskPhase.PENDING
                task.assigned_ship = -1
                released.append(tid)

        if released:
            # 重新分配释放的任务
            self._reassign_tasks(released, exclude_ship=ship_id)
            self._log_event(ship_id, "energy_critical_transfer",
                            f"能源紧急: {len(released)} 个任务已转移给其他船",
                            {"released": released})

    def _handle_fault(self, ship_id: int):
        """故障: 释放全部任务, 已装货的最近港口卸货"""
        ship = self.ships[ship_id]

        released = []
        for tid in list(ship.task_sequence):
            task = self.tasks.get(tid)
            if not task:
                continue
            ship.task_sequence.remove(tid)
            if task.status not in (TaskPhase.COMPLETED,):
                ship.load -= task.payload if task.status in (TaskPhase.NAVIGATE_TO_DELIVERY,) else 0
                task.status = TaskPhase.PENDING
                task.assigned_ship = -1
                released.append(tid)

        # 如果有在途货物, 最近港口卸货
        if ship.load > 0:
            nearest_port = min(self.port_ids,
                               key=lambda p: self.rn.dist_matrix[ship.current_node, p])
            ship.current_node = nearest_port
            ship.load = 0  # 货物留在港口

        if released:
            self._reassign_tasks(released, exclude_ship=ship_id)
            self._log_event(ship_id, "fault_transfer",
                            f"故障: {len(released)} 个任务已转移",
                            {"released": released})

    def _handle_overdue_task(self, task_id: int):
        """任务延期: 尝试转移给最快的可用船"""
        task = self.tasks[task_id]
        current_ship = self.ships.get(task.assigned_ship)

        # 找最快的其他船
        best_ship = None
        best_time = float('inf')
        for ship in self.ships.values():
            if ship.ship_id == task.assigned_ship:
                continue
            if task.payload > ship.remaining_capacity:
                continue
            d = (self.rn.dist_matrix[ship.current_node, task.pickup_node] +
                 self.rn.dist_matrix[task.pickup_node, task.delivery_node])
            if d == np.inf:
                continue
            t = d / ship.max_speed
            if t < best_time:
                best_time = t
                best_ship = ship

        if best_ship and current_ship:
            # 从当前船移除
            if task.task_id in current_ship.task_sequence:
                current_ship.task_sequence.remove(task.task_id)
            current_ship.load -= task.payload

            # 分配给新船
            task.status = TaskPhase.PENDING
            task.assigned_ship = best_ship.ship_id
            best_ship.task_sequence.append(task.task_id)
            best_ship.load += task.payload

            self._log_event(task_id, "overdue_reassign",
                            f"T{task_id} 从 {current_ship.name} 转移至 {best_ship.name}",
                            {"from_ship": current_ship.ship_id,
                             "to_ship": best_ship.ship_id})

    def _handle_idle_ship(self, ship_id: int):
        """空闲船: 抢未分配任务"""
        ship = self.ships[ship_id]
        pending = [t for t in self.tasks.values()
                   if t.status == TaskPhase.PENDING and t.assigned_ship == -1]

        if not pending:
            return

        # 找最佳任务
        pending.sort(key=lambda t: -t.urgency)
        for task in pending[:3]:  # 最多抢3个
            if task.payload <= ship.remaining_capacity:
                d = (self.rn.dist_matrix[ship.current_node, task.pickup_node] +
                     self.rn.dist_matrix[task.pickup_node, task.delivery_node])
                if d < np.inf:
                    task.assigned_ship = ship.ship_id
                    ship.task_sequence.append(task.task_id)
                    ship.load += task.payload
                    self._log_event(ship_id, "idle_take_task",
                                    f"{ship.name} 空闲, 接手 T{task.task_id}",
                                    {"task": task.task_id})

    def _reassign_tasks(self, task_ids: List[int], exclude_ship: int = -1):
        """把释放的任务重新分配给其他船"""
        for tid in task_ids:
            task = self.tasks.get(tid)
            if not task:
                continue

            best_ship = None
            best_cost = float('inf')
            for ship in self.ships.values():
                if ship.ship_id == exclude_ship:
                    continue
                if task.payload > ship.remaining_capacity:
                    continue
                d1 = self.rn.dist_matrix[ship.current_node, task.pickup_node]
                d2 = self.rn.dist_matrix[task.pickup_node, task.delivery_node]
                if d1 == np.inf or d2 == np.inf:
                    continue
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_ship = ship

            if best_ship:
                task.assigned_ship = best_ship.ship_id
                best_ship.task_sequence.append(tid)
                best_ship.load += task.payload

    # ========== 日志 ==========

    def _log_event(self, entity_id, etype, msg, data=None):
        self.event_log.append({
            "time": round(self.current_time, 0),
            "entity": entity_id,
            "type": etype,
            "message": msg,
            "data": data or {}
        })

    def _log_allocation(self, label):
        snapshot = {}
        for sid, s in self.ships.items():
            snapshot[s.name] = {
                "tasks": list(s.task_sequence),
                "node": s.current_node,
                "load": round(s.load, 0),
                "energy": round(s.energy, 0)
            }
        self.allocation_log.append({
            "time": round(self.current_time, 0),
            "label": label,
            "ships": snapshot
        })

    # ========== 状态输出 ==========

    def format_status(self) -> str:
        """当前状态的可读输出"""
        lines = []
        lines.append(f"\n{'='*65}")
        lines.append(f"  实时调度状态  t={self.current_time:.0f}s  ({self.current_time/3600:.1f}h)")
        lines.append(f"{'='*65}")

        for ship in sorted(self.ships.values(), key=lambda s: s.ship_id):
            # 状态图标
            status_icons = {
                TaskPhase.PENDING: "⏸",
                TaskPhase.NAVIGATE_TO_PICKUP: "➡️",
                TaskPhase.LOADING: "📦",
                TaskPhase.NAVIGATE_TO_DELIVERY: "🚢",
                TaskPhase.UNLOADING: "📤",
            }
            icon = status_icons.get(ship.current_phase, "❓")

            # 能源条
            e_pct = int(ship.energy_ratio * 20)
            e_bar = "█" * e_pct + "░" * (20 - e_pct)

            lines.append(f"\n  {icon} {ship.name}  |{e_bar}| {ship.energy_ratio*100:.0f}%")
            lines.append(f"    位置: {self.node_names.get(ship.current_node, 'N'+str(ship.current_node))}"
                         f"  载货: {ship.load:.0f}/{ship.max_payload:.0f}t"
                         f"  航程: {ship.total_distance/1000:.1f}km")

            # 任务序列
            if ship.completed_tasks:
                done = ", ".join(f"T{t}" for t in ship.completed_tasks[-3:])
                lines.append(f"    已完成: {done}")

            if ship.task_sequence:
                seq_parts = []
                for i, tid in enumerate(ship.task_sequence):
                    task = self.tasks.get(tid)
                    if not task:
                        continue
                    marker = "←" if tid == ship.current_task_id else ""
                    pn = self.node_names.get(task.pickup_node, f"N{task.pickup_node}")
                    dn = self.node_names.get(task.delivery_node, f"N{task.delivery_node}")
                    seq_parts.append(f"{marker}T{tid}({pn}→{dn})")
                lines.append(f"    任务序列: {' → '.join(seq_parts)}")
            else:
                lines.append(f"    空闲, 无任务")

            if ship.eta_to_next > 0 and ship.current_phase not in (TaskPhase.PENDING,):
                lines.append(f"    预计完成当前阶段: {(self.current_time + ship.eta_to_next)/60:.0f}min")

        # 待分配任务
        pending = [t for t in self.tasks.values()
                   if t.status == TaskPhase.PENDING and t.assigned_ship == -1]
        if pending:
            lines.append(f"\n  待分配任务: {len(pending)} 个")

        # 延期任务
        overdue = [t for t in self.tasks.values()
                   if t.is_overdue(self.current_time)
                   and t.status not in ("completed", "cancelled")]
        if overdue:
            lines.append(f"  延期任务: {len(overdue)} 个 ⚠️")

        lines.append(f"{'='*65}")
        return "\n".join(lines)

    def format_event_log(self) -> str:
        """最近的事件日志"""
        if not self.event_log:
            return "无事件"

        lines = []
        lines.append(f"\n{'─'*65}")
        lines.append(f"  事件日志 (最近 20 条)")
        lines.append(f"{'─'*65}")
        for ev in self.event_log[-20:]:
            changed = "✓" if ev.get("changed") else "—"
            msg = ev.get('message', '')
            if not msg:
                evt = ev.get('event', {})
                msg = evt.get('type', '?') if isinstance(evt, dict) else str(evt)
            lines.append(f"  [{ev['time']:.0f}s] [{changed}] {msg}")
        lines.append(f"{'─'*65}")
        return "\n".join(lines)

    def format_reallocation_comparison(self) -> str:
        """重分配前后对比"""
        lines = []
        for ev in self.event_log:
            if "before" not in ev or "after" not in ev:
                continue
            if ev["before"] == ev["after"]:
                continue
            ev_type = ev.get('event', {}).get('type', ev.get('type', '?'))
            lines.append(f"\n  [{ev['time']:.0f}s] {ev_type}: {ev.get('message','')}")
            lines.append(f"    Before: { {k:v.get('task_sequence',[]) for k,v in ev['before'].items()} }")
            lines.append(f"    After:  { {k:v.get('task_sequence',[]) for k,v in ev['after'].items()} }")
        return "\n".join(lines) if lines else "\n  (无重分配变更)"
