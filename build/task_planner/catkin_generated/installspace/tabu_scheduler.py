#!/usr/bin/env python3
"""
Tabu Search — 禁忌表+渴望准则 (精简版)
核心: 禁忌记忆防止回头, 每步采样候选
"""
import random
from copy import deepcopy
from typing import Dict, List
from base_scheduler import BaseScheduler, RouteNode


class TabuScheduler(BaseScheduler):

    def __init__(self, road_network, ships, tasks):
        super().__init__(road_network, ships, tasks)
        self.max_iter = 300
        self.sample_size = 80  # 每轮采样候选数
        self.tabu_tenure = 12

    def optimize(self) -> Dict[int, List[RouteNode]]:
        routes = self.greedy_init()
        best_routes = deepcopy(routes)
        best_ms = self.makespan(best_routes)

        current_routes = deepcopy(routes)
        tabu = {}  # (tid, old_ship) → remaining_tenure

        for it in range(self.max_iter):
            tasks = self._collect_all_tasks(current_routes)
            if len(tasks) < 1:
                continue

            best_candidate, best_candidate_ms = None, float('inf')
            best_key = None

            for _ in range(self.sample_size):
                # 随机选一个任务 + 目标船
                sid_from, tid = random.choice(tasks)
                other = [s for s in self.ships if s != sid_from]
                if not other: continue
                sid_to = random.choice(other)

                move_key = (tid, sid_from)
                if tabu.get(move_key, 0) > 0:
                    continue  # 禁忌

                new_routes = self._copy_routes(current_routes)
                self._remove_task(new_routes, sid_from, tid)
                if not self._greedy_insert_one(new_routes, sid_to, tid):
                    continue
                if not self._validate_all(new_routes):
                    continue

                ms = self.makespan(new_routes)

                # 渴望准则: 优于全局最优
                if ms < best_ms:
                    best_candidate = new_routes
                    best_candidate_ms = ms
                    best_key = move_key
                    break

                if ms < best_candidate_ms:
                    best_candidate = new_routes
                    best_candidate_ms = ms
                    best_key = move_key

            if best_candidate is None:
                continue

            current_routes = best_candidate
            if best_candidate_ms < best_ms:
                best_routes = deepcopy(best_candidate)
                best_ms = best_candidate_ms

            if best_key:
                tabu[best_key] = self.tabu_tenure

            # 衰减禁忌
            expired = [k for k, v in tabu.items() if v <= 0]
            for k in expired:
                del tabu[k]
            for k in list(tabu.keys()):
                tabu[k] -= 1

        return best_routes

    def _greedy_insert_one(self, routes, sid, tid):
        task = self.tasks.get(tid)
        if task is None: return False
        pu, de, ok = self._find_insertion(self.ships[sid], routes[sid], task)
        if ok:
            self._do_insert(routes, sid, tid, pu, de, task)
        return ok
