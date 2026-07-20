#!/usr/bin/env python3
"""One planning facade used by both initial allocation and rolling re-planning."""
from typing import Dict, Iterable, Set
from hetero_graph import build_hetero_graph
from graph_evaluator import GraphEvaluator
from alns_scheduler import ALNSScheduler
from route_builder import plans_from_route_nodes, expand_plan
from domain import ActionType, RouteAction


class GraphALNSPlanner:
    def __init__(self, road_network, node_names=None, max_iter=400, k_candidates=5,
                 gas_ids=None, refuel_time=600.0, reserve_ratio=0.10,
                 use_learned_bottleneck=False, use_multi_start=True,
                 alns_overrides=None):
        self.rn = road_network
        self.node_names = node_names or {}
        self.max_iter = max_iter
        self.k_candidates = k_candidates
        self.gas_ids = list(gas_ids or [])
        self.refuel_time = refuel_time
        self.reserve_ratio = reserve_ratio
        # The learned candidate ranking is retained.  The learned destroy
        # operator is opt-in: its current offline result is worse than the
        # standard ALNS destroy/repair portfolio on the static benchmark.
        self.use_learned_bottleneck = use_learned_bottleneck
        # Compete a learned HGT-biased construction with a deterministic
        # cost/balance construction before spending the ALNS budget.  Both
        # produce the identical RouteNode interface used by rolling replans.
        self.use_multi_start = use_multi_start
        self.alns_overrides = dict(alns_overrides or {})
        self.candidate_scorer = None
        self.version = 0
        self.last_graph = None

    def plan(self, ships: Dict, tasks: Dict, now=0.0, mutable_tasks: Set[int] = None,
             existing_plans: Dict = None):
        """Plan only mutable, not-yet-started tasks; executing tasks stay frozen."""
        active_ships = {sid: s for sid, s in ships.items() if not s.failed}
        selected = {tid: t for tid, t in tasks.items()
                    if t.status in ('pending', 'assigned', 'at_transfer')
                    and (mutable_tasks is None or tid in mutable_tasks)}
        self.last_graph = build_hetero_graph(active_ships, selected, self.rn)
        evaluator = GraphEvaluator(self.rn, self.node_names, selected)
        candidate_map = (self.candidate_scorer.rank(self.last_graph)
                         if self.candidate_scorer else self._graph_candidate_map(active_ships, selected))
        order_scores = self.candidate_scorer.last_confidence if self.candidate_scorer else {}
        alns_config = {
            'use_graph_candidates': True, 'use_adaptive_weights': True,
            'use_sa': True, 'max_iter': self.max_iter, 'k_candidates': self.k_candidates,
            'graph_candidate_map': candidate_map,
            'task_order_scores': order_scores,
            'learned_bottleneck': self.use_learned_bottleneck,
        }
        alns_config.update(self.alns_overrides)
        alns = ALNSScheduler(evaluator, selected, self.rn, self.node_names, config=alns_config)
        if self.candidate_scorer and getattr(self.candidate_scorer, 'enabled', False):
            learned_routes = alns.build_learned_initial_routes(active_ships,
                self.candidate_scorer.last_ranked_scores,
                self.candidate_scorer.last_confidence)
            routes = learned_routes
            self.last_initial_strategy = 'hgt'
            if self.use_multi_start:
                cost_routes = alns.build_initial_routes(active_ships)
                # Lexicographic selection keeps the static primary objective
                # explicit instead of relying on a scale-dependent aggregate.
                def key(candidate):
                    raw = alns._fleet_cost_raw(candidate, active_ships)
                    return (raw['M'], raw['D'], raw['E'])
                if key(cost_routes) < key(learned_routes):
                    routes = cost_routes
                    self.last_initial_strategy = 'cost_balance'
        else:
            routes = alns.build_initial_routes(active_ships)
            self.last_initial_strategy = 'cost_balance'
        routes = alns.optimize(active_ships, routes)
        self.version += 1
        plans = plans_from_route_nodes(self.rn, active_ships, selected, routes, self.version, now)
        return {sid: self._insert_refuels(active_ships[sid], plan, selected) for sid, plan in plans.items()}

    def _graph_candidate_map(self, ships, tasks):
        """Use hetero Ship--Task edge features, not only raw nearest distance."""
        candidate_map = {tid: [] for tid in tasks}
        for tid in tasks:
            j = self.last_graph.task_idx[tid]
            ranked = []
            for k in range(self.last_graph.st_edges.shape[1]):
                si, tj = self.last_graph.st_edges[:, k]
                if tj == j:
                    # feature[6] is graph match score; higher is better
                    ranked.append((float(self.last_graph.st_feat[k, 6]), self.last_graph.ship_ids[int(si)]))
            candidate_map[tid] = [sid for _, sid in sorted(ranked, reverse=True)[:self.k_candidates]]
        return candidate_map

    def _insert_refuels(self, ship, plan, tasks):
        """Insert a nearest reachable REFUEL action before violating the reserve."""
        if not self.gas_ids:
            return plan
        actions, current, energy, load = [], ship.current_node, ship.energy, ship.load
        for action in plan.actions:
            d = self.rn.dist_matrix[current, action.node_id]
            need = d / 1000.0 * ship.energy_per_km * (1.0 + 0.5 * load / max(ship.max_payload, 1))
            if energy - need < ship.max_energy * self.reserve_ratio:
                choices = [(self.rn.dist_matrix[current, gid], gid) for gid in self.gas_ids]
                choices = [(dist, gid) for dist, gid in choices if dist < float('inf')]
                if choices:
                    _, gid = min(choices)
                    actions.append(RouteAction(ActionType.REFUEL, gid, -1, self.refuel_time))
                    current, energy = gid, ship.max_energy
                    d = self.rn.dist_matrix[current, action.node_id]
                    need = d / 1000.0 * ship.energy_per_km * (1.0 + 0.5 * load / max(ship.max_payload, 1))
            actions.append(action)
            energy -= need
            if action.action == ActionType.PICKUP:
                load += tasks[action.task_id].payload
            elif action.action == ActionType.DELIVERY:
                load -= tasks[action.task_id].payload
            current = action.node_id
        plan.actions = actions
        return expand_plan(self.rn, ship, plan)
