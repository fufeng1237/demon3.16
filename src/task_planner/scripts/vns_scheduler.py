#!/usr/bin/env python3
"""
VNS — 变邻域搜索 (精简版)
核心: 3邻域系统切换 + 扰动跳出局部最优
"""
import random, numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from base_scheduler import BaseScheduler, RouteNode


class VNSScheduler(BaseScheduler):

    def __init__(self, road_network, ships, tasks):
        super().__init__(road_network, ships, tasks)
        self.max_iter = 200
        self.sample_size = 50   # 每个邻域随机采样候选数

    def optimize(self) -> Dict[int, List[RouteNode]]:
        routes = self.greedy_init()
        best_routes = deepcopy(routes)
        best_ms = self.makespan(best_routes)

        for it in range(self.max_iter):
            improved = False
            for k in [1, 2, 3]:
                new_routes = self._sample_neighborhood(routes, k, self.sample_size)
                if new_routes is None:
                    continue
                new_ms = self.makespan(new_routes)
                if new_ms < best_ms - 1.0:
                    routes = new_routes
                    best_routes = deepcopy(new_routes)
                    best_ms = new_ms
                    improved = True
                    break  # 回到 N1

            if not improved:
                routes = self._shaking(best_routes)

        return best_routes

    def _sample_neighborhood(self, routes, k, n_samples):
        """在邻域 k 中随机采样 n_samples 个候选, 返回最优"""
        tasks = self._collect_all_tasks(routes)
        if len(tasks) < 1:
            return None

        best_new, best_ms = None, float('inf')

        for _ in range(n_samples):
            new_routes = self._copy_routes(routes)

            if k == 1:  # Swap
                if len(tasks) < 2: continue
                a = random.choice(tasks)
                b = random.choice([t for t in tasks if t[0] != a[0]])
                self._remove_task(new_routes, a[0], a[1])
                self._remove_task(new_routes, b[0], b[1])
                ok1 = self._greedy_insert_one(new_routes, a[0], b[1])
                ok2 = self._greedy_insert_one(new_routes, b[0], a[1])
                if not (ok1 and ok2 and self._validate_all(new_routes)): continue

            elif k == 2:  # Relocate
                a = random.choice(tasks)
                other = [s for s in self.ships if s != a[0]]
                if not other: continue
                target = random.choice(other)
                self._remove_task(new_routes, a[0], a[1])
                if not self._greedy_insert_one(new_routes, target, a[1]): continue
                if not self._validate_all(new_routes): continue

            elif k == 3:  # Intra-2Opt
                sid = random.choice(list(self.ships.keys()))
                seq = new_routes.get(sid, [])
                if len(seq) < 4: continue
                i = random.randint(0, len(seq) - 2)
                j = random.randint(i + 1, len(seq) - 1)
                new_seq = seq[:i] + list(reversed(seq[i:j+1])) + seq[j+1:]
                if not self._check_route(self.ships[sid], new_seq): continue
                new_routes[sid] = new_seq

            else:
                continue

            ms = self.makespan(new_routes)
            if ms < best_ms:
                best_ms = ms
                best_new = new_routes

        return best_new

    def _shaking(self, routes):
        """随机 destroy 20% + 贪心 repair"""
        tasks = self._collect_all_tasks(routes)
        n = max(1, int(len(tasks) * 0.2))
        to_remove = random.sample(tasks, min(n, len(tasks)))
        new_routes = self._copy_routes(routes)
        removed_info = []
        for sid, tid in to_remove:
            removed_info.append((tid, sid))
            self._remove_task(new_routes, sid, tid)
        # 乱序重新插入 (不保证插入原船)
        random.shuffle(removed_info)
        for tid, _ in removed_info:
            self._greedy_insert_best(new_routes, tid)
        return new_routes

    def _greedy_insert_one(self, routes, sid, tid):
        task = self.tasks.get(tid)
        if task is None: return False
        pu, de, ok = self._find_insertion(self.ships[sid], routes[sid], task)
        if ok:
            self._do_insert(routes, sid, tid, pu, de, task)
        return ok

    def _greedy_insert_best(self, routes, tid):
        task = self.tasks.get(tid)
        if task is None: return
        best_sid, best_pu, best_de, best_score = None, -1, -1, float('inf')
        for sid in self.ships:
            ship = self.ships[sid]
            pu, de, ok = self._find_insertion(ship, routes[sid], task)
            if not ok: continue
            dt = self._insertion_time_delta(ship, routes[sid], pu, de, task)
            st = self._ship_time(ship, routes[sid])
            score = dt + 0.5 * st
            if score < best_score:
                best_score = score; best_sid = sid; best_pu, best_de = pu, de
        if best_sid is not None:
            self._do_insert(routes, best_sid, tid, best_pu, best_de, task)
