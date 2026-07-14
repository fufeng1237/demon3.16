#!/usr/bin/env python3
"""
GA — 遗传算法 (精简版)
核心: 种群进化, 交叉+变异+选择
"""
import random
from copy import deepcopy
from typing import Dict, List
from base_scheduler import BaseScheduler, RouteNode


class GAScheduler(BaseScheduler):

    def __init__(self, road_network, ships, tasks):
        super().__init__(road_network, ships, tasks)
        self.pop_size = 20
        self.n_generations = 100
        self.mutation_rate = 0.4

    def optimize(self) -> Dict[int, List[RouteNode]]:
        greedy = self.greedy_init()

        # 初始化种群
        population = [greedy]
        for _ in range(self.pop_size - 1):
            population.append(self._mutate(deepcopy(greedy)))

        fitness = [self.makespan(ind) for ind in population]

        for gen in range(self.n_generations):
            new_pop = []

            # 精英保留 top 2
            elite_idx = sorted(range(len(fitness)), key=lambda i: fitness[i])[:2]
            for i in elite_idx:
                new_pop.append(deepcopy(population[i]))

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitness)
                p2 = self._tournament(population, fitness)
                child = self._crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                if self._validate_all(child):
                    new_pop.append(child)
                else:
                    new_pop.append(deepcopy(greedy))  # 退化到贪心

            population = new_pop[:self.pop_size]
            fitness = [self.makespan(ind) for ind in population]

        best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        return population[best_idx]

    def _tournament(self, pop, fitness, k=3):
        indices = random.sample(range(len(pop)), min(k, len(pop)))
        return pop[min(indices, key=lambda i: fitness[i])]

    def _crossover(self, p1, p2):
        """
        交叉: 继承父A某一艘船的全部任务, 其余船从父B继承
        """
        child = {sid: [] for sid in self.ships}
        donor_ship = random.choice(list(self.ships.keys()))

        # 父A的 donor_ship 任务 → 子代
        donor_tasks = set()
        for rn in p1.get(donor_ship, []):
            if rn.action == "PICKUP":
                donor_tasks.add(rn.task_id)

        # 其余船从父B继承 (排除 donor_tasks)
        all_assigned = set(donor_tasks)
        for sid in self.ships:
            if sid == donor_ship: continue
            for rn in p2.get(sid, []):
                if rn.action == "PICKUP" and rn.task_id not in all_assigned:
                    task = self.tasks.get(rn.task_id)
                    if task:
                        pu, de, ok = self._find_insertion(self.ships[sid], child[sid], task)
                        if ok:
                            self._do_insert(child, sid, rn.task_id, pu, de, task)
                            all_assigned.add(rn.task_id)

        # 插入 donor_tasks
        for tid in donor_tasks:
            self._greedy_insert_one(child, donor_ship, tid)

        # 插入遗漏任务
        for tid in self.tasks:
            if tid not in all_assigned:
                self._greedy_insert_best(child, tid)

        return child

    def _mutate(self, routes):
        """变异: 从最多任务的船移一个到最少任务的船"""
        new_routes = self._copy_routes(routes)
        ship_counts = {}
        for sid, seq in new_routes.items():
            ship_counts[sid] = len([r for r in seq if r.action == "PICKUP"])

        max_s = max(ship_counts, key=ship_counts.get)
        min_s = min(ship_counts, key=ship_counts.get)
        if max_s == min_s:
            return new_routes

        max_tasks = [rn.task_id for rn in new_routes[max_s] if rn.action == "PICKUP"]
        if not max_tasks:
            return new_routes

        tid = random.choice(max_tasks)
        self._remove_task(new_routes, max_s, tid)
        if not self._greedy_insert_one(new_routes, min_s, tid):
            self._greedy_insert_one(new_routes, max_s, tid)

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
