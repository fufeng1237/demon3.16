#!/usr/bin/env python3
"""
Memetic Algorithm — 模因算法 (精简版)
核心: GA 种群进化 + 局部搜索教育 (LS as education operator)

论文出处:
  Nalepa, J. & Blocho, M. (2017).
  "A Parallel Memetic Algorithm for the Pickup and Delivery Problem with Time Windows."
  25th Euromicro International Conference on Parallel, Distributed and
  Network-based Processing (PDP), pp. 1–8.

  Yuskov, A.D., Kulachenko, I.N., Kochetov, Y.A. (2025).
  "Hybrid Memetic Algorithm for the Pickup and Delivery Problem with Time Windows."
  Mathematical Optimization Theory and Operations Research (MOTOR 2025).
  该论文在 53 个 Li & Lim 基准实例上改进了已知最优解。

算法原理:
  1. 种群进化框架: 选择 → 交叉 → 变异 → 局部搜索教育
  2. 关键创新: 每个后代在加入种群前必须经过局部搜索"教育"
     (这是 Memetic 与普通 GA 的核心区别)
  3. 局部搜索: swap/relocate 邻域迭代，直到无法改进
  4. 精英保留 + 锦标赛选择 + 路径保留交叉

区分特征 (vs GA):
  - GA: 交叉+变异后直接入种群，无局部搜索
  - Memetic: 交叉+变异后必须做局部搜索 (LS as education)
  - 效果: 收敛速度更快，解质量更高，但每代计算量更大
"""
import random
from copy import deepcopy
from typing import Dict, List
from base_scheduler import BaseScheduler, RouteNode


class MemeticScheduler(BaseScheduler):

    def __init__(self, road_network, ships, tasks):
        super().__init__(road_network, ships, tasks)
        self.pop_size = 16
        self.n_generations = 60
        self.mutation_rate = 0.5
        self.ls_iterations = 30  # 每个后代的局部搜索步数

    def optimize(self) -> Dict[int, List[RouteNode]]:
        greedy = self.greedy_init()

        # ── 初始化种群 ──
        population = [greedy]
        for _ in range(self.pop_size - 1):
            mutated = deepcopy(greedy)
            mutated = self._random_disrupt(mutated)
            mutated = self._local_search(mutated)
            population.append(mutated)

        fitness = [self.makespan(ind) for ind in population]

        for gen in range(self.n_generations):
            new_pop = []

            # 精英保留 top 2
            elite_idx = sorted(range(len(fitness)), key=lambda i: fitness[i])[:2]
            for i in elite_idx:
                new_pop.append(deepcopy(population[i]))

            while len(new_pop) < self.pop_size:
                # 父代选择
                p1 = self._tournament(population, fitness, k=3)
                p2 = self._tournament(population, fitness, k=3)

                # 交叉
                child = self._crossover(p1, p2)

                # 变异
                if random.random() < self.mutation_rate:
                    child = self._random_disrupt(child)

                # ★ 模因核心: 局部搜索教育 (LS as education operator)
                child = self._local_search(child)

                # 退化保护
                if not self._validate_all(child):
                    child = deepcopy(greedy)

                new_pop.append(child)

            population = new_pop[:self.pop_size]
            fitness = [self.makespan(ind) for ind in population]

        best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        return population[best_idx]

    # ================================================================
    #  局部搜索 (Memetic 核心)
    # ================================================================

    def _local_search(self, routes, n_iter=None):
        """
        局部搜索教育: 对给定个体进行迭代改善
        使用 relocate 和 swap 两个邻域, 贪心接受改进
        """
        if n_iter is None:
            n_iter = self.ls_iterations

        best = deepcopy(routes)
        best_ms = self.makespan(best)

        for _ in range(n_iter):
            tasks = self._collect_all_tasks(best)
            if len(tasks) < 1:
                continue

            improved = False

            # ── Relocate 邻域 ──
            for _ in range(10):
                sid_from, tid = random.choice(tasks)
                other = [s for s in self.ships if s != sid_from]
                if not other:
                    continue
                sid_to = random.choice(other)

                new_routes = self._copy_routes(best)
                self._remove_task(new_routes, sid_from, tid)
                task = self.tasks.get(tid)
                if task is None:
                    continue

                pu, de, ok = self._find_insertion(self.ships[sid_to],
                                                   new_routes[sid_to], task)
                if not ok:
                    continue

                self._do_insert(new_routes, sid_to, tid, pu, de, task)
                if not self._validate_all(new_routes):
                    continue

                ms = self.makespan(new_routes)
                if ms < best_ms:
                    best = new_routes
                    best_ms = ms
                    improved = True
                    break  # first-improvement

            if improved:
                continue

            # ── Swap 邻域 ──
            for _ in range(10):
                if len(tasks) < 2:
                    break
                a = random.choice(tasks)
                b_candidates = [t for t in tasks if t[0] != a[0]]
                if not b_candidates:
                    continue
                b = random.choice(b_candidates)

                new_routes = self._copy_routes(best)
                self._remove_task(new_routes, a[0], a[1])
                self._remove_task(new_routes, b[0], b[1])

                ta, tb = self.tasks.get(a[1]), self.tasks.get(b[1])
                if ta is None or tb is None:
                    continue

                ok1 = self._greedy_insert_one(new_routes, a[0], b[1])
                ok2 = self._greedy_insert_one(new_routes, b[0], a[1])
                if not (ok1 and ok2):
                    continue
                if not self._validate_all(new_routes):
                    continue

                ms = self.makespan(new_routes)
                if ms < best_ms:
                    best = new_routes
                    best_ms = ms
                    improved = True
                    break

            if not improved:
                break  # 局部最优

        return best

    # ================================================================
    #  遗传算子
    # ================================================================

    def _tournament(self, pop, fitness, k=3):
        """锦标赛选择"""
        indices = random.sample(range(len(pop)), min(k, len(pop)))
        return pop[min(indices, key=lambda i: fitness[i])]

    def _crossover(self, p1, p2):
        """
        路径保留交叉 (Route-Preserving Crossover):
          继承父A某一艘船的全部任务, 其余船从父B继承
          这保证了部分优秀路径完整保留
        """
        child = {sid: [] for sid in self.ships}
        donor_ship = random.choice(list(self.ships.keys()))

        # 父A donor_ship 的任务 → 子代
        donor_tasks = set()
        for rn in p1.get(donor_ship, []):
            if rn.action == "PICKUP":
                donor_tasks.add(rn.task_id)

        # 其余船从父B继承 (排除 donor_tasks)
        all_assigned = set(donor_tasks)
        for sid in self.ships:
            if sid == donor_ship:
                continue
            for rn in p2.get(sid, []):
                if rn.action == "PICKUP" and rn.task_id not in all_assigned:
                    task = self.tasks.get(rn.task_id)
                    if task:
                        pu, de, ok = self._find_insertion(self.ships[sid],
                                                           child[sid], task)
                        if ok:
                            self._do_insert(child, sid, rn.task_id, pu, de, task)
                            all_assigned.add(rn.task_id)

        # 插入 donor_tasks 到 donor_ship
        for tid in donor_tasks:
            self._greedy_insert_one(child, donor_ship, tid)

        # 插入遗漏任务
        for tid in self.tasks:
            if tid not in all_assigned:
                self._greedy_insert_best(child, tid)

        return child

    def _random_disrupt(self, routes):
        """
        随机扰动 (替代简单变异): 强制从最多任务的船搬 n 个到最少任务的船
        """
        new_routes = self._copy_routes(routes)

        ship_counts = {}
        for sid, seq in new_routes.items():
            ship_counts[sid] = len([r for r in seq if r.action == "PICKUP"])
        max_s = max(ship_counts, key=ship_counts.get)
        min_s = min(ship_counts, key=ship_counts.get)
        if max_s == min_s:
            return new_routes

        # 搬 n_move 个任务
        max_tasks = [rn.task_id for rn in new_routes[max_s]
                     if rn.action == "PICKUP"]
        n_move = min(2, len(max_tasks))
        if n_move == 0:
            return new_routes

        to_move = random.sample(max_tasks, n_move)
        for tid in to_move:
            self._remove_task(new_routes, max_s, tid)
            task = self.tasks.get(tid)
            if task is None:
                continue
            ok = self._greedy_insert_one(new_routes, min_s, tid)
            if not ok:
                # failback: 插回原船
                self._greedy_insert_one(new_routes, max_s, tid)

        return new_routes

    # ================================================================
    #  辅助方法
    # ================================================================

    def _greedy_insert_one(self, routes, sid, tid):
        task = self.tasks.get(tid)
        if task is None:
            return False
        pu, de, ok = self._find_insertion(self.ships[sid], routes[sid], task)
        if ok:
            self._do_insert(routes, sid, tid, pu, de, task)
        return ok

    def _greedy_insert_best(self, routes, tid):
        task = self.tasks.get(tid)
        if task is None:
            return
        best_sid, best_pu, best_de, best_score = None, -1, -1, float('inf')
        for sid in self.ships:
            ship = self.ships[sid]
            pu, de, ok = self._find_insertion(ship, routes[sid], task)
            if not ok:
                continue
            dt = self._insertion_time_delta(ship, routes[sid], pu, de, task)
            st = self._ship_time(ship, routes[sid])
            score = dt + 0.5 * st
            if score < best_score:
                best_score = score
                best_sid, best_pu, best_de = sid, pu, de
        if best_sid is not None:
            self._do_insert(routes, best_sid, tid, best_pu, best_de, task)
