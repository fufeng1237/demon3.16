#!/usr/bin/env python3
"""
状态评估器 — Ship 中心决策

Ship 状态 (6维):
  [0] energy_score     = remaining / needed_for_remaining
  [1] load_ratio       = current_load / max_payload
  [2] health           = 0~1
  [3] idle             = 0/1
  [4] position_score   = dist_to_target / expected_dist
  [5] eta_reliability  = actual_elapsed / estimated_time

决策: release_all | release_some | take_tasks | normal
"""

import numpy as np


def evaluate_ship(ship, tasks, road_network, current_time):
    """
    评估单艘船的状态 → 返回 (action, reason, state_dict)

    Task.urgency → 影响 energy_score (紧急任务需要更多能源)
    Road.congestion → 影响 eta_reliability (拥堵导致延误)
    """
    state = {}

    # ── [0] energy_score = 剩余 / 所需 ──
    needed = 0.0
    for tid in ship.task_sequence:
        task = tasks.get(tid)
        if not task or task.status in ('completed', 'cancelled'):
            continue
        d = (road_network.dist_matrix[ship.current_node, task.pickup_node] +
             road_network.dist_matrix[task.pickup_node, task.delivery_node])
        base_energy = d / 1000.0 * ship.energy_per_km

        # Task urgency 影响: 紧急任务加权
        if task.deadline < float('inf'):
            remaining_t = max(60, task.deadline - current_time)
            travel_t = d / max(ship.max_speed, 1)
            urgency_factor = min(3.0, travel_t / remaining_t)
            base_energy *= (1.0 + urgency_factor)

        needed += base_energy

    state['energy_score'] = ship.energy / max(1.0, needed)
    state['energy_detail'] = f"{ship.energy:.0f}/{needed:.0f}kWh"

    # ── [1] load_ratio ──
    state['load_ratio'] = ship.load / max(ship.max_payload, 1)

    # ── [2] health ──
    state['health'] = ship.health

    # ── [3] idle ──
    state['idle'] = 1.0 if (not ship.task_sequence and ship.load == 0) else 0.0

    # ── [4] position_score ──
    if ship.current_task_id >= 0 and ship.current_phase == 'navigate_to_pickup':
        task = tasks.get(ship.current_task_id)
        if task:
            target_dist = road_network.dist_matrix[ship.current_node, task.pickup_node]
            # 预期航行距离 vs 已航行距离
            if target_dist > 0 and ship.total_distance > 0:
                # 越接近 1.0 = 正常, >1.2 = 偏航
                expected = target_dist * (ship.total_time / max(1, ship.total_time))
                state['position_score'] = max(0.5, min(2.0, ship.total_distance / max(1, target_dist)))
            else:
                state['position_score'] = 1.0
        else:
            state['position_score'] = 1.0
    elif ship.current_phase == 'navigate_to_delivery':
        task = tasks.get(ship.current_task_id)
        if task:
            target_dist = road_network.dist_matrix[ship.current_node, task.delivery_node]
            state['position_score'] = min(2.0, ship.total_distance / max(1, target_dist)) if target_dist > 0 else 1.0
        else:
            state['position_score'] = 1.0
    else:
        state['position_score'] = 1.0

    # ── [5] eta_reliability ──
    elapsed = ship.total_time
    estimated = 0.0
    for tid in ship.task_sequence:
        task = tasks.get(tid)
        if not task or task.status in ('completed',):
            continue
        d = road_network.dist_matrix[ship.current_node, task.pickup_node]
        estimated += d / max(ship.max_speed, 1) + 300  # +5min loading

    # Road congestion 影响
    congestion_factor = 1.0
    node = road_network.nodes.get(ship.current_node)
    if node and hasattr(node, 'congestion'):
        congestion_factor = max(1.0, 1.0 + getattr(node, 'congestion', 0) * 0.5)

    if estimated > 0 and elapsed > 0:
        state['eta_reliability'] = min(5.0, (elapsed / estimated) * congestion_factor)
    else:
        state['eta_reliability'] = 1.0

    state['eta_detail'] = f"elapsed={elapsed:.0f}s, est={estimated:.0f}s, cong={congestion_factor:.1f}"

    # ── 决策 ──
    if state['energy_score'] < 0.5:
        return ("release_all", "能源不足以完成剩余任务", state)
    if state['health'] < 0.3:
        return ("release_all", "健康度不足, 故障风险", state)

    if state['energy_score'] < 0.8:
        n = max(1, int(len(ship.task_sequence) * (1 - state['energy_score'])))
        return ("release_some", f"能源紧张(score={state['energy_score']:.2f}), 释放{n}个", state)
    if state['eta_reliability'] > 1.5:
        return ("release_some", f"严重延误(eta={state['eta_reliability']:.1f}), 转移下游任务", state)

    if state['idle'] > 0 and state['health'] > 0.7 and state['energy_score'] > 0.8:
        return ("take_tasks", "空闲且健康, 可接新任务", state)

    return ("normal", "", state)


def evaluate_fleet(ships, tasks, road_network, current_time):
    """
    评估整个船队, 返回汇总的决策.
    """
    decisions = {}
    release_pool = []   # 释放的任务
    idle_ships = []     # 空闲可接任务的船

    for sid, ship in ships.items():
        action, reason, state = evaluate_ship(ship, tasks, road_network, current_time)
        decisions[sid] = (action, reason, state)

        if action == "release_all":
            for tid in list(ship.task_sequence):
                task = tasks.get(tid)
                if task and task.status not in ('completed', 'cancelled'):
                    task.status = 'pending'
                    task.assigned_ship = -1
                    release_pool.append(tid)
                if tid in ship.task_sequence:
                    ship.task_sequence.remove(tid)
            ship.load = 0

        elif action == "release_some":
            n_release = max(1, int(len(ship.task_sequence) * 0.3))
            released = 0
            for tid in reversed(list(ship.task_sequence)):
                if released >= n_release:
                    break
                task = tasks.get(tid)
                if task and task.status in ('pending', 'assigned'):
                    task.status = 'pending'
                    task.assigned_ship = -1
                    release_pool.append(tid)
                    ship.task_sequence.remove(tid)
                    ship.load = max(0, ship.load - task.payload)
                    released += 1

        elif action == "take_tasks":
            idle_ships.append(sid)

    return decisions, release_pool, idle_ships
