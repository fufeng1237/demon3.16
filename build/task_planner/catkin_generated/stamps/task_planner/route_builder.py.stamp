#!/usr/bin/env python3
"""Expand high-level pickup/delivery actions into road-node sequences."""
import heapq
from typing import Dict, List
from domain import ActionType, RouteAction, RoutePlan, ShipState, TransportTask


def shortest_path(rn, start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]
    queue, dist, prev = [(0.0, start)], {start: 0.0}, {}
    while queue:
        d, u = heapq.heappop(queue)
        if d != dist.get(u):
            continue
        if u == goal:
            break
        for v in rn.adj.get(u, []):
            nd = d + float(rn.dist_matrix[u, v])
            if nd < dist.get(v, float('inf')):
                dist[v], prev[v] = nd, u
                heapq.heappush(queue, (nd, v))
    if goal not in dist:
        raise ValueError(f'road network cannot reach {start} -> {goal}')
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return list(reversed(path))


def expand_plan(rn, ship: ShipState, plan: RoutePlan) -> RoutePlan:
    current = ship.current_node
    for action in plan.actions:
        action.road_nodes = shortest_path(rn, current, action.node_id)
        current = action.node_id
    return plan


def plans_from_route_nodes(rn, ships: Dict[int, ShipState], tasks: Dict[int, TransportTask],
                           routes: Dict, version: int, created_at: float = 0.0) -> Dict[int, RoutePlan]:
    """Convert legacy-compatible RouteNode lists to versioned executable plans."""
    plans = {}
    for sid, nodes in routes.items():
        actions = []
        for node in nodes:
            if node.action == 'PICKUP':
                actions.append(RouteAction(ActionType.PICKUP, node.node_id, node.task_id, 300.0))
            elif node.action == 'DELIVERY':
                actions.append(RouteAction(ActionType.DELIVERY, node.node_id, node.task_id, 180.0))
        plans[sid] = expand_plan(rn, ships[sid], RoutePlan(sid, version, actions, created_at=created_at))
    return plans
